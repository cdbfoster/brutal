use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::mem;
use std::path::Path;
use std::str::FromStr;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

const WORD_LENGTH: usize = 5;
const GUESS_LIMIT: usize = 6;

#[derive(Copy, Clone, Debug, PartialEq)]
struct Word([char; WORD_LENGTH]);

impl ToString for Word {
    fn to_string(&self) -> String {
        self.0.iter().collect()
    }
}

impl FromStr for Word {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() == WORD_LENGTH {
            let mut word = Word(['_'; WORD_LENGTH]);
            word.0
                .iter_mut()
                .zip(value.chars())
                .for_each(|(d, c)| *d = c);
            Ok(word)
        } else {
            Err("word has incorrect length")
        }
    }
}

fn read_lines(filename: impl AsRef<Path>) -> Vec<Word> {
    let file = File::open(filename).expect("no such file");
    let buf = BufReader::new(file);
    buf.lines()
        .map(|l| l.expect("could not parse line"))
        .map(|s| Word::from_str(&s).expect("could not parse word"))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Constraint {
    /// Present, and in the correct place.
    Green(char, usize),
    /// Present, but not in the correct place.
    Yellow(char, usize),
    /// Not present.
    Gray(char),
}

fn get_constraints(answer: &Word, guess: &Word, buffer: &mut Vec<Constraint>) {
    buffer.clear();

    let mut answer = *answer;
    let mut guess = *guess;

    for (i, c) in guess.0.iter_mut().enumerate() {
        if answer.0[i] == *c {
            buffer.push(Constraint::Green(*c, i));
            answer.0[i] = '_';
            *c = '_';
        }
    }

    for (i, c) in guess.0.iter_mut().enumerate().filter(|(_, c)| **c != '_') {
        if let Some(j) = answer.0.iter().position(|d| d == c) {
            buffer.push(Constraint::Yellow(*c, i));
            answer.0[j] = '_';
            *c = '_';
        }
    }

    for c in guess.0.iter().filter(|c| **c != '_') {
        if !buffer.contains(&Constraint::Gray(*c)) {
            buffer.push(Constraint::Gray(*c));
        }
    }
}

fn passes_constraint(word: &Word, constraint: &Constraint) -> bool {
    match constraint {
        Constraint::Green(c, i) => word.0[*i] == *c,
        Constraint::Yellow(c, i) => word.0.contains(c) && word.0[*i] != *c,
        Constraint::Gray(c) => !word.0.contains(c),
    }
}

fn passes_constraints(word: &Word, constraints: &[Constraint]) -> bool {
    let mut characters = *word;

    for constraint in constraints {
        if !passes_constraint(&characters, constraint) {
            return false;
        }

        match constraint {
            Constraint::Green(_, i) => characters.0[*i] = '_',
            Constraint::Yellow(c, _) => *characters.0.iter_mut().find(|d| *d == c).unwrap() = '_',
            Constraint::Gray(_) => (),
        }
    }

    true
}

fn filter_word_list(words: &[Word], constraints: &[Constraint], buffer: &mut Vec<Word>) {
    buffer.clear();

    words
        .iter()
        .filter(|w| passes_constraints(w, constraints))
        .for_each(|w| buffer.push(*w));
}

fn get_score(
    answer: &Word,
    guess: &Word,
    words: &[Word],
    starting_guess: usize,
    constraint_buffers: &mut [Vec<Constraint>],
    word_buffers: &mut [Vec<Word>],
) -> (f32, f32) {
    if answer == guess {
        return (starting_guess as f32, 1.0);
    }

    if starting_guess >= GUESS_LIMIT {
        return (0.0, 0.0);
    }

    get_constraints(answer, guess, &mut constraint_buffers[0]);
    filter_word_list(words, &constraint_buffers[0], &mut word_buffers[0]);

    let (_, next_constraint_buffers) = constraint_buffers.split_at_mut(1);
    let (next_words, next_word_buffers) = word_buffers.split_at_mut(1);

    let mut guesses_sum = 0.0;
    let mut success_sum = 0.0;

    for word in next_words[0].iter() {
        let (guess_count, success_rate) = get_score(
            answer,
            word,
            &next_words[0],
            starting_guess + 1,
            next_constraint_buffers,
            next_word_buffers,
        );

        guesses_sum += guess_count * success_rate;
        success_sum += success_rate;
    }

    if success_sum > 0.0 {
        (
            guesses_sum / success_sum,
            success_sum / next_words[0].len() as f32,
        )
    } else {
        (0.0, 0.0)
    }
}

#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(
        short,
        long,
        help = "Override the default answer list",
        default_value = "wordle_answer_list.txt"
    )]
    answer_list: String,

    #[clap(
        short,
        long,
        help = "Override the default guess list [defaults to the answer list]"
    )]
    guess_list: Option<String>,

    #[clap(
        short,
        long,
        help = "Override the default search list [defaults to the guess list]"
    )]
    search_list: Option<String>,

    #[clap(
        short,
        long,
        parse(try_from_str),
        help = "Add <WORD> to the guess list and replace the search list with <WORD> unless the search list is explicity set"
    )]
    word: Vec<Word>,

    #[clap(
        short,
        long,
        help = "Override the default output path",
        default_value = "word_scores.csv"
    )]
    output_file: String,

    #[clap(short, long, default_value = "1")]
    threads: usize,
}

fn main() {
    let args = Args::parse();

    let answer_words = read_lines(&args.answer_list);
    let mut guess_words = read_lines(&args.guess_list.unwrap_or(args.answer_list));
    let mut search_words = if let Some(search_list) = &args.search_list {
        read_lines(search_list)
    } else {
        guess_words.clone()
    };

    if !args.word.is_empty() {
        guess_words.extend_from_slice(&args.word);
        if args.search_list.is_some() {
            search_words.extend_from_slice(&args.word);
        } else {
            search_words = args.word;
        }
    }

    let answer_words = Arc::new(answer_words);
    let guess_words = Arc::new(guess_words);

    println!("Word counts:");
    println!("  Possible answers:  {:5}", answer_words.len());
    println!("  Available guesses: {:5}", guess_words.len());
    println!("  Words to search:   {:5}", search_words.len());
    println!();

    let search_queue = Arc::new(Mutex::new(
        search_words.iter().rev().copied().collect::<Vec<_>>(),
    ));

    let progress_bars = MultiProgress::new();
    let progress_bar_style =
        ProgressStyle::default_bar().template("{elapsed_precise} {bar:50} {pos:>5}/{len:>5} {msg}");

    let (completed, completed_receiver) = mpsc::channel();

    let worker_threads = (0..args.threads)
        .map(|_| {
            let answer_words = answer_words.clone();
            let guess_words = guess_words.clone();
            let search_queue = search_queue.clone();

            let progress = progress_bars.add(ProgressBar::new(answer_words.len() as u64));
            progress.set_style(progress_bar_style.clone());
            progress.enable_steady_tick(500);

            let completed = completed.clone();

            thread::spawn(move || {
                // So we don't slap the shit out of the heap with our search.
                let mut constraint_buffers = vec![Vec::with_capacity(WORD_LENGTH); GUESS_LIMIT];
                let mut word_buffers = vec![Vec::with_capacity(guess_words.len()); GUESS_LIMIT];

                while let Some(guess) = {
                    let mut search_queue_guard = search_queue.lock().unwrap();
                    search_queue_guard.pop()
                } {
                    progress.reset();

                    let mut guesses_sum = 0.0;
                    let mut success_sum = 0.0;

                    for answer in answer_words.iter() {
                        progress.set_message(format!(
                            "{} -> {}",
                            guess.to_string(),
                            answer.to_string()
                        ));

                        let (guess_count, success_rate) = get_score(
                            answer,
                            &guess,
                            &guess_words,
                            1,
                            &mut constraint_buffers,
                            &mut word_buffers,
                        );

                        progress.inc(1);

                        guesses_sum += guess_count * success_rate;
                        success_sum += success_rate;
                    }

                    let guess_count = if success_sum > 0.0 {
                        guesses_sum / success_sum
                    } else {
                        0.0
                    };

                    let success_rate = success_sum / answer_words.len() as f32;

                    completed
                        .send((guess, (guess_count, success_rate)))
                        .expect("could not send update");
                }

                progress.finish_with_message("done");
            })
        })
        .collect::<Vec<_>>();

    mem::drop(completed);

    let total_progress = progress_bars.add(ProgressBar::new(search_words.len() as u64));
    total_progress.set_style(progress_bar_style);
    total_progress.enable_steady_tick(500);

    let progress_thread = thread::spawn(move || progress_bars.join().unwrap());

    let collection_thread = thread::spawn(move || {
        let mut word_scores = Vec::with_capacity(search_words.len());

        while let Ok((word, (guess_count, success_rate))) = completed_receiver.recv() {
            total_progress.inc(1);
            total_progress.set_message(format!(
                "{}, average: {:.3}, success: {:5.2}%",
                word.to_string(),
                guess_count,
                success_rate * 100.0,
            ));

            word_scores.push((word, (guess_count, success_rate)));
            word_scores.sort_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap());

            let mut file = File::create(&args.output_file).expect("cannot open output file");

            writeln!(file, "{:w$} average, success", "word,", w = WORD_LENGTH + 1)
                .expect("cannot write header");

            for (word, (guess_count, success_rate)) in word_scores.iter() {
                writeln!(
                    file,
                    "{}, {:7.3}, {:7.4}",
                    word.to_string(),
                    guess_count,
                    success_rate
                )
                .expect("cannot write line");
            }
        }
        total_progress.finish_with_message("done");
    });

    worker_threads.into_iter().for_each(|t| t.join().unwrap());
    collection_thread.join().unwrap();
    progress_thread.join().unwrap();
}
