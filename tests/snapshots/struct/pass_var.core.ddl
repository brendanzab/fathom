//! Test referring to aliases in struct fields.

struct Pair {
    first : primitive U8,
    second : primitive U8,
}

MyPair = item Pair;

struct PairPair {
    first : item Pair,
    second : item MyPair,
}
