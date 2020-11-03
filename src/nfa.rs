use std::collections::HashSet;

use self::CharacterClass::{Ascii, InvalidChars, ValidChars};

/// A set of characters
#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct CharSet {
    low_mask: u64,
    high_mask: u64,
    non_ascii: HashSet<char>,
}

impl CharSet {
    /// Create a new CharSet
    pub fn new() -> Self {
        Self {
            low_mask: 0,
            high_mask: 0,
            non_ascii: HashSet::new(),
        }
    }

    /// Insert a character into the CharSet
    pub fn insert(&mut self, char: char) {
        let val = char as u32 - 1;

        if val > 127 {
            self.non_ascii.insert(char);
        } else if val > 63 {
            let bit = 1 << (val - 64);
            self.high_mask |= bit;
        } else {
            let bit = 1 << val;
            self.low_mask |= bit;
        }
    }

    /// Determine of `char` exists in this CharSet
    pub fn contains(&self, char: char) -> bool {
        let val = char as u32 - 1;

        if val > 127 {
            self.non_ascii.contains(&char)
        } else if val > 63 {
            let bit = 1 << (val - 64);
            self.high_mask & bit != 0
        } else {
            let bit = 1 << val;
            self.low_mask & bit != 0
        }
    }
}

/// A group of either valid or invalid characters.
///
/// These are equivalent to the `[]` groups in regex
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum CharacterClass {
    Ascii(u64, u64, bool),
    ValidChars(CharSet),
    InvalidChars(CharSet),
}

impl CharacterClass {
    /// Any character is valid
    pub fn any() -> Self {
        Ascii(u64::max_value(), u64::max_value(), true)
    }

    /// Characters in the input `string` are valid matches.
    /// Equivalent to the regex `[<string>]`
    pub fn valid(string: &str) -> Self {
        ValidChars(Self::str_to_set(string))
    }

    /// Characters in the input `string` are invalid.
    /// Equivalent to the regex `[^<string>]`
    pub fn invalid(string: &str) -> Self {
        InvalidChars(Self::str_to_set(string))
    }

    /// Only the given `char` is valid.
    /// Equivalent to the regex `<char>`.
    pub fn valid_char(char: char) -> Self {
        let val = char as u32 - 1;

        if val > 127 {
            ValidChars(Self::char_to_set(char))
        } else if val > 63 {
            Ascii(1 << (val - 64), 0, false)
        } else {
            Ascii(0, 1 << val, false)
        }
    }

    /// Only the given `char` is invalid.
    /// Equivalent to the regex `[^<char>]`
    pub fn invalid_char(char: char) -> Self {
        let val = char as u32 - 1;

        if val > 127 {
            InvalidChars(Self::char_to_set(char))
        } else if val > 63 {
            Ascii(u64::max_value() ^ (1 << (val - 64)), u64::max_value(), true)
        } else {
            Ascii(u64::max_value(), u64::max_value() ^ (1 << val), true)
        }
    }

    /// Checks whether the given `char` is allowed in the set
    pub fn matches(&self, char: char) -> bool {
        match *self {
            ValidChars(ref valid) => valid.contains(char),
            InvalidChars(ref invalid) => !invalid.contains(char),
            Ascii(high, low, unicode) => {
                let val = char as u32 - 1;
                if val > 127 {
                    unicode
                } else if val > 63 {
                    high & (1 << (val - 64)) != 0
                } else {
                    low & (1 << val) != 0
                }
            }
        }
    }

    /// Utility function to turn a char into a `CharSet`
    fn char_to_set(char: char) -> CharSet {
        let mut set = CharSet::new();
        set.insert(char);
        set
    }

    /// Utility function to turn a string into a `CharSet`
    fn str_to_set(string: &str) -> CharSet {
        let mut set = CharSet::new();
        for char in string.chars() {
            set.insert(char);
        }
        set
    }
}

/// A thread that has weaved through our NFA
#[derive(Debug, Clone)]
struct Thread {
    state: usize,
    captures: Vec<(usize, usize)>,
    capture_begin: Option<usize>,
}

impl Thread {
    /// Create a new Thread at state 0
    pub fn new() -> Self {
        Self {
            state: 0,
            captures: Vec::new(),
            capture_begin: None,
        }
    }

    /// Start a capture
    #[inline]
    pub fn start_capture(&mut self, start: usize) {
        self.capture_begin = Some(start);
    }

    /// Finish a capture
    #[inline]
    pub fn end_capture(&mut self, end: usize) {
        self.captures.push((self.capture_begin.unwrap(), end));
        self.capture_begin = None;
    }

    /// Extract all captures
    pub fn extract<'a>(&self, source: &'a str) -> Vec<&'a str> {
        self.captures
            .iter()
            .map(|&(begin, end)| &source[begin..end])
            .collect()
    }
}

/// A state for the non-deterministic finite automaton to be in
#[derive(Debug, Clone)]
pub struct State<T> {
    /// The index of this state
    pub index: usize,
    /// The character class of this state
    pub chars: CharacterClass,
    /// The states that we can move to
    pub next_states: Vec<usize>,
    /// Can we finish here?
    pub acceptance: bool,
    /// Do we start a capture here?
    pub start_capture: bool,
    /// Do we end a capture here?
    pub end_capture: bool,
    /// Storing metadata in our states
    pub metadata: Option<T>,
}

impl<T> PartialEq for State<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> State<T> {
    /// Create a new state at an index with some chars
    pub fn new(index: usize, chars: CharacterClass) -> Self {
        Self {
            index,
            chars,
            next_states: Vec::new(),
            acceptance: false,
            start_capture: false,
            end_capture: false,
            metadata: None,
        }
    }
}

/// A successful match of the NFA with the final state and its captures
#[derive(Debug)]
pub struct Match<'a> {
    pub state: usize,
    pub captures: Vec<&'a str>,
}

impl<'a> Match<'a> {
    /// Create the successful match from a state and captures
    pub fn new<'b>(state: usize, captures: Vec<&'b str>) -> Match<'b> {
        Match { state, captures }
    }
}

/// An error that indicates an issue processing a string with the NFA
#[derive(Debug)]
pub struct ProcessFailure(String);

impl core::fmt::Display for ProcessFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::write!(f, "Couldn't process {}", self.0)
    }
}
impl std::error::Error for ProcessFailure {}

/// An error that indicates that an acceptance state was not reached whilst processing a string
#[derive(Debug)]
pub struct Exhausted(String);

impl core::fmt::Display for Exhausted {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::write!(
            f,
            "{} was exhausted before reaching an acceptance state",
            self.0
        )
    }
}
impl std::error::Error for Exhausted {}

/// An error whilst processing a string using the NFA
#[derive(Debug)]
pub enum NFAError {
    ProcessFailure(ProcessFailure),
    Exhausted(Exhausted),
}

impl core::fmt::Display for NFAError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::write!(
            f,
            "{}",
            match self {
                Self::ProcessFailure(err) => err.to_string(),
                Self::Exhausted(err) => err.to_string(),
            }
        )
    }
}
impl std::error::Error for NFAError {}

impl core::convert::From<ProcessFailure> for NFAError {
    fn from(err: ProcessFailure) -> Self {
        Self::ProcessFailure(err)
    }
}

impl core::convert::From<Exhausted> for NFAError {
    fn from(err: Exhausted) -> Self {
        Self::Exhausted(err)
    }
}

/// A structure representing a non-deterministic finite automaton
#[derive(Debug, Clone, Default)]
pub struct NFA<T> {
    /// The list of possible states
    states: Vec<State<T>>,
    /// Whether each state can start a capture
    start_capture: Vec<bool>,
    /// Whether each state can end a capture
    end_capture: Vec<bool>,
    /// Does this state indicate acceptance?
    acceptance: Vec<bool>,
}

impl<T> NFA<T> {
    /// Create a new automaton
    pub fn new() -> Self {
        let root = State::new(0, CharacterClass::any());
        Self {
            states: vec![root],
            start_capture: vec![false],
            end_capture: vec![false],
            acceptance: vec![false],
        }
    }

    /// Try to match a string with the NFA and a function that gives orderings to the states.
    ///
    /// If a match is found, it is returned with its captures and final state, otherwise an
    /// `NFAError` is returned.
    /// An `NFAError` indicates that either an acceptance state couldn't be reached, or there was a
    /// failure in processing the string.
    pub fn process<'a, I, F>(&self, string: &'a str, mut ord: F) -> Result<Match<'a>, NFAError>
    where
        I: Ord,
        F: FnMut(usize) -> I,
    {
        let mut threads = vec![Thread::new()];

        for (i, char) in string.chars().enumerate() {
            // Try the next character
            let next_threads = self.process_char(threads, char, i);

            // Nowhere to go
            if next_threads.is_empty() {
                return Err(ProcessFailure(string.into()).into());
            }

            threads = next_threads;
        }

        // Only threads that are at acceptance
        let returned = threads
            .into_iter()
            .filter(|thread| self.get(thread.state).acceptance);

        // Try to reduce the threads to just one
        let thread = returned
            .fold(None, |prev, y| {
                let y_v = ord(y.state);
                match prev {
                    None => Some((y_v, y)),
                    Some((x_v, x)) => {
                        if x_v < y_v {
                            Some((y_v, y))
                        } else {
                            Some((x_v, x))
                        }
                    }
                }
            })
            .map(|p| p.1);

        match thread {
            None => Err(Exhausted(string.into()).into()),
            Some(mut thread) => {
                if thread.capture_begin.is_some() {
                    thread.end_capture(string.len());
                }
                let state = self.get(thread.state);
                Ok(Match::new(state.index, thread.extract(string)))
            }
        }
    }

    /// Process a singular character through some number of canditate threads
    #[inline]
    fn process_char(&self, threads: Vec<Thread>, char: char, pos: usize) -> Vec<Thread> {
        let mut returned = Vec::with_capacity(threads.len());

        for mut thread in threads {
            let current_state = self.get(thread.state);

            // PERFORMANCE:
            // Update the state in place if only one future state matches
            // This avoids extra allocations
            {
                let mut count = 0;
                let mut found_state = 0;

                for &index in &current_state.next_states {
                    let state = &self.states[index];

                    if state.chars.matches(char) {
                        count += 1;
                        found_state = index;
                    }
                }

                if count == 1 {
                    thread.state = found_state;
                    capture(self, &mut thread, current_state.index, found_state, pos);
                    returned.push(thread);
                    continue;
                }
            }

            // Add all other states
            for &index in &current_state.next_states {
                let state = &self.states[index];
                if state.chars.matches(char) {
                    let mut thread = fork_thread(&thread, state);
                    capture(self, &mut thread, current_state.index, index, pos);
                    returned.push(thread);
                }
            }
        }

        returned
    }

    /// Get a state at a position immutably
    #[inline]
    pub fn get(&self, state: usize) -> &State<T> {
        &self.states[state]
    }

    /// Get a state at a position mutably
    pub fn get_mut(&mut self, state: usize) -> &mut State<T> {
        &mut self.states[state]
    }

    /// Add an edge from a state `index` to a character class
    pub fn put(&mut self, index: usize, chars: CharacterClass) -> usize {
        {
            let state = self.get(index);

            for &index in &state.next_states {
                let state = self.get(index);
                if state.chars == chars {
                    return index;
                }
            }
        }

        let state = self.new_state(chars);
        self.get_mut(index).next_states.push(state);
        state
    }

    /// Add an edge from a state to another state
    pub fn put_state(&mut self, index: usize, child: usize) {
        if !self.states[index].next_states.contains(&child) {
            self.get_mut(index).next_states.push(child);
        }
    }

    /// Indicate that this state can be accepted
    pub fn acceptance(&mut self, index: usize) {
        self.get_mut(index).acceptance = true;
        self.acceptance[index] = true;
    }

    /// Indicate that this state starts a capture
    pub fn start_capture(&mut self, index: usize) {
        self.get_mut(index).start_capture = true;
        self.start_capture[index] = true;
    }

    /// Indicate that this state ends a capture
    pub fn end_capture(&mut self, index: usize) {
        self.get_mut(index).end_capture = true;
        self.end_capture[index] = true;
    }

    /// Add some metadata to a state
    pub fn metadata(&mut self, index: usize, metadata: T) {
        self.get_mut(index).metadata = Some(metadata);
    }

    /// Create a new state from a character class
    fn new_state(&mut self, chars: CharacterClass) -> usize {
        let index = self.states.len();
        let state = State::new(index, chars);
        self.states.push(state);

        self.acceptance.push(false);
        self.start_capture.push(false);
        self.end_capture.push(false);

        index
    }
}

/// Create a cloned thread with modified state
#[inline]
fn fork_thread<T>(thread: &Thread, state: &State<T>) -> Thread {
    let mut new_trace = thread.clone();
    new_trace.state = state.index;
    new_trace
}

/// Start an end captures on a thread if we need to
#[inline]
fn capture<T>(
    nfa: &NFA<T>,
    thread: &mut Thread,
    current_state: usize,
    next_state: usize,
    pos: usize,
) {
    if thread.capture_begin == None && nfa.start_capture[next_state] {
        thread.start_capture(pos);
    }

    if thread.capture_begin != None && nfa.end_capture[current_state] && next_state > current_state
    {
        thread.end_capture(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::{CharSet, CharacterClass, NFA};

    #[test]
    fn basic_test() {
        let mut nfa = NFA::<()>::new();
        let a = nfa.put(0, CharacterClass::valid("h"));
        let b = nfa.put(a, CharacterClass::valid("e"));
        let c = nfa.put(b, CharacterClass::valid("l"));
        let d = nfa.put(c, CharacterClass::valid("l"));
        let e = nfa.put(d, CharacterClass::valid("o"));
        nfa.acceptance(e);

        let m = nfa.process("hello", |a| a);

        assert!(
            m.unwrap().state == e,
            "You didn't get the right final state"
        );
    }

    #[test]
    fn multiple_solutions() {
        let mut nfa = NFA::<()>::new();
        let a1 = nfa.put(0, CharacterClass::valid("n"));
        let b1 = nfa.put(a1, CharacterClass::valid("e"));
        let c1 = nfa.put(b1, CharacterClass::valid("w"));
        nfa.acceptance(c1);

        let a2 = nfa.put(0, CharacterClass::invalid(""));
        let b2 = nfa.put(a2, CharacterClass::invalid(""));
        let c2 = nfa.put(b2, CharacterClass::invalid(""));
        nfa.acceptance(c2);

        let m = nfa.process("new", |a| a);

        assert!(m.unwrap().state == c2, "The two states were not found");
    }

    #[test]
    fn multiple_paths() {
        let mut nfa = NFA::<()>::new();
        let a = nfa.put(0, CharacterClass::valid("t")); // t
        let b1 = nfa.put(a, CharacterClass::valid("h")); // th
        let c1 = nfa.put(b1, CharacterClass::valid("o")); // tho
        let d1 = nfa.put(c1, CharacterClass::valid("m")); // thom
        let e1 = nfa.put(d1, CharacterClass::valid("a")); // thoma
        let f1 = nfa.put(e1, CharacterClass::valid("s")); // thomas

        let b2 = nfa.put(a, CharacterClass::valid("o")); // to
        let c2 = nfa.put(b2, CharacterClass::valid("m")); // tom

        nfa.acceptance(f1);
        nfa.acceptance(c2);

        let thomas = nfa.process("thomas", |a| a);
        let tom = nfa.process("tom", |a| a);
        let thom = nfa.process("thom", |a| a);
        let nope = nfa.process("nope", |a| a);

        assert!(thomas.unwrap().state == f1, "thomas was parsed correctly");
        assert!(tom.unwrap().state == c2, "tom was parsed correctly");
        assert!(thom.is_err(), "thom didn't reach an acceptance state");
        assert!(nope.is_err(), "nope wasn't parsed");
    }

    #[test]
    fn repetitions() {
        let mut nfa = NFA::<()>::new();
        let a = nfa.put(0, CharacterClass::valid("p")); // p
        let b = nfa.put(a, CharacterClass::valid("o")); // po
        let c = nfa.put(b, CharacterClass::valid("s")); // pos
        let d = nfa.put(c, CharacterClass::valid("t")); // post
        let e = nfa.put(d, CharacterClass::valid("s")); // posts
        let f = nfa.put(e, CharacterClass::valid("/")); // posts/
        let g = nfa.put(f, CharacterClass::invalid("/")); // posts/[^/]
        nfa.put_state(g, g);

        nfa.acceptance(g);

        let post = nfa.process("posts/1", |a| a);
        let new_post = nfa.process("posts/new", |a| a);
        let invalid = nfa.process("posts/", |a| a);

        assert!(post.unwrap().state == g, "posts/1 was parsed");
        assert!(new_post.unwrap().state == g, "posts/new was parsed");
        assert!(invalid.is_err(), "posts/ was invalid");
    }

    #[test]
    fn repetitions_with_ambiguous() {
        let mut nfa = NFA::<()>::new();
        let a = nfa.put(0, CharacterClass::valid("p")); // p
        let b = nfa.put(a, CharacterClass::valid("o")); // po
        let c = nfa.put(b, CharacterClass::valid("s")); // pos
        let d = nfa.put(c, CharacterClass::valid("t")); // post
        let e = nfa.put(d, CharacterClass::valid("s")); // posts
        let f = nfa.put(e, CharacterClass::valid("/")); // posts/
        let g1 = nfa.put(f, CharacterClass::invalid("/")); // posts/[^/]
        let g2 = nfa.put(f, CharacterClass::valid("n")); // posts/n
        let h2 = nfa.put(g2, CharacterClass::valid("e")); // posts/ne
        let i2 = nfa.put(h2, CharacterClass::valid("w")); // posts/new

        nfa.put_state(g1, g1);

        nfa.acceptance(g1);
        nfa.acceptance(i2);

        let post = nfa.process("posts/1", |a| a);
        let ambiguous = nfa.process("posts/new", |a| a);
        let invalid = nfa.process("posts/", |a| a);

        assert!(post.unwrap().state == g1, "posts/1 was parsed");
        assert!(ambiguous.unwrap().state == i2, "posts/new was ambiguous");
        assert!(invalid.is_err(), "posts/ was invalid");
    }

    #[test]
    fn captures() {
        let mut nfa = NFA::<()>::new();
        let a = nfa.put(0, CharacterClass::valid("n"));
        let b = nfa.put(a, CharacterClass::valid("e"));
        let c = nfa.put(b, CharacterClass::valid("w"));

        nfa.acceptance(c);
        nfa.start_capture(a);
        nfa.end_capture(c);

        let post = nfa.process("new", |a| a);

        assert_eq!(post.unwrap().captures, vec!["new"]);
    }

    #[test]
    fn capture_mid_match() {
        let mut nfa = NFA::<()>::new();
        let a = nfa.put(0, valid('p'));
        let b = nfa.put(a, valid('/'));
        let c = nfa.put(b, invalid('/'));
        let d = nfa.put(c, valid('/'));
        let e = nfa.put(d, valid('c'));

        nfa.put_state(c, c);
        nfa.acceptance(e);
        nfa.start_capture(c);
        nfa.end_capture(c);

        let post = nfa.process("p/123/c", |a| a);

        assert_eq!(post.unwrap().captures, vec!["123"]);
    }

    #[test]
    fn capture_multiple_captures() {
        let mut nfa = NFA::<()>::new();
        let a = nfa.put(0, valid('p'));
        let b = nfa.put(a, valid('/'));
        let c = nfa.put(b, invalid('/'));
        let d = nfa.put(c, valid('/'));
        let e = nfa.put(d, valid('c'));
        let f = nfa.put(e, valid('/'));
        let g = nfa.put(f, invalid('/'));

        nfa.put_state(c, c);
        nfa.put_state(g, g);
        nfa.acceptance(g);

        nfa.start_capture(c);
        nfa.end_capture(c);

        nfa.start_capture(g);
        nfa.end_capture(g);

        let post = nfa.process("p/123/c/456", |a| a);
        assert_eq!(post.unwrap().captures, vec!["123", "456"]);
    }

    #[test]
    fn test_ascii_set() {
        let mut set = CharSet::new();
        set.insert('?');
        set.insert('a');
        set.insert('é');

        assert!(set.contains('?'), "The set contains char 63");
        assert!(set.contains('a'), "The set contains char 97");
        assert!(set.contains('é'), "The set contains char 233");
        assert!(!set.contains('q'), "The set does not contain q");
        assert!(!set.contains('ü'), "The set does not contain ü");
    }

    fn valid(char: char) -> CharacterClass {
        CharacterClass::valid_char(char)
    }

    fn invalid(char: char) -> CharacterClass {
        CharacterClass::invalid_char(char)
    }
}
