// This file is automatically @generated by fathom 0.1.0
// It is not intended for manual editing.

//! Test referring to aliases in struct fields.

#[derive(Copy, Clone)]
pub struct Pair {
    first: u8,
    second: u8,
}

impl Pair {
    pub fn first(&self) -> u8 {
        self.first
    }

    pub fn second(&self) -> u8 {
        self.second
    }
}

impl fathom_runtime::Format for Pair {
    type Host = Pair;
}

impl<'data> fathom_runtime::ReadFormat<'data> for Pair {
    fn read(reader: &mut fathom_runtime::FormatReader<'data>) -> Result<Pair, fathom_runtime::ReadError> {
        let first = reader.read::<fathom_runtime::U8>()?;
        let second = reader.read::<fathom_runtime::U8>()?;

        Ok(Pair {
            first,
            second,
        })
    }
}

pub type MyPair = Pair;

#[derive(Copy, Clone)]
pub struct PairPair {
    first: Pair,
    second: Pair,
}

impl PairPair {
    pub fn first(&self) -> Pair {
        self.first
    }

    pub fn second(&self) -> Pair {
        self.second
    }
}

impl fathom_runtime::Format for PairPair {
    type Host = PairPair;
}

impl<'data> fathom_runtime::ReadFormat<'data> for PairPair {
    fn read(reader: &mut fathom_runtime::FormatReader<'data>) -> Result<PairPair, fathom_runtime::ReadError> {
        let first = reader.read::<Pair>()?;
        let second = reader.read::<Pair>()?;

        Ok(PairPair {
            first,
            second,
        })
    }
}
