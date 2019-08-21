//! The core type theory of the data description language.

use codespan::{ByteIndex, FileId, Span};
use codespan_reporting::diagnostic::Diagnostic;
use pretty::{DocAllocator, DocBuilder};
use std::borrow::Borrow;
use std::fmt;
use std::sync::Arc;

use crate::diagnostics;
use crate::lexer::SpannedToken;

mod grammar {
    include!(concat!(env!("OUT_DIR"), "/core/grammar.rs"));
}

pub mod semantics;
pub mod validate;

/// A label.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label(pub String);

impl Label {
    pub fn doc<'core, D>(&'core self, alloc: &'core D) -> DocBuilder<'core, D>
    where
        D: DocAllocator<'core>,
        D::Doc: Clone,
    {
        alloc.text(&self.0)
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Borrow<str> for Label {
    fn borrow(&self) -> &str {
        &self.0
    }
}

/// A module of items.
#[derive(Debug, Clone)]
pub struct Module {
    /// The file in which this module was defined.
    pub file_id: FileId,
    /// The items in this module.
    pub items: Vec<Item>,
}

impl Module {
    pub fn parse(
        file_id: FileId,
        tokens: impl IntoIterator<Item = Result<SpannedToken, Diagnostic>>,
        report: &mut dyn FnMut(Diagnostic),
    ) -> Module {
        grammar::ModuleParser::new()
            .parse(file_id, report, tokens)
            .unwrap_or_else(|error| {
                report(diagnostics::error::parse(file_id, error));
                Module {
                    file_id,
                    items: Vec::new(),
                }
            })
    }

    pub fn doc<'core, D>(&'core self, alloc: &'core D) -> DocBuilder<'core, D>
    where
        D: DocAllocator<'core>,
        D::Doc: Clone,
    {
        let items = alloc.intersperse(
            self.items.iter().map(|item| item.doc(alloc)),
            alloc.newline().append(alloc.newline()),
        );

        items.append(alloc.newline())
    }
}

impl PartialEq for Module {
    fn eq(&self, other: &Module) -> bool {
        self.items == other.items
    }
}

/// Items in a module.
#[derive(Debug, Clone)]
pub enum Item {
    /// Alias definitions
    Alias(Alias),
    /// Struct definitions.
    Struct(StructType),
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::Struct(struct_ty) => struct_ty.span,
            Item::Alias(alias) => alias.span,
        }
    }

    pub fn doc<'core, D>(&'core self, alloc: &'core D) -> DocBuilder<'core, D>
    where
        D: DocAllocator<'core>,
        D::Doc: Clone,
    {
        match self {
            Item::Alias(alias) => alias.doc(alloc),
            Item::Struct(struct_ty) => struct_ty.doc(alloc),
        }
    }
}

impl PartialEq for Item {
    fn eq(&self, other: &Item) -> bool {
        match (self, other) {
            (Item::Alias(alias0), Item::Alias(alias1)) => *alias0 == *alias1,
            (Item::Struct(struct_ty0), Item::Struct(struct_ty1)) => *struct_ty0 == *struct_ty1,
            (_, _) => false,
        }
    }
}

/// A struct type definition.
#[derive(Debug, Clone)]
pub struct Alias {
    /// The full span of this definition.
    pub span: Span,
    /// Doc comment.
    pub doc: Arc<[String]>,
    /// Name of this definition.
    pub name: Label,
    /// The term that is aliased.
    pub term: Term,
}

impl Alias {
    pub fn doc<'core, D>(&'core self, alloc: &'core D) -> DocBuilder<'core, D>
    where
        D: DocAllocator<'core>,
        D::Doc: Clone,
    {
        let docs = alloc.concat(self.doc.iter().map(|line| {
            (alloc.nil())
                .append("///")
                .append(line)
                .group()
                .append(alloc.newline())
        }));

        (alloc.nil())
            .append(docs)
            .append(self.name.doc(alloc))
            .append(alloc.space())
            .append("=")
            .group()
            .append(
                (alloc.nil())
                    .append(alloc.space())
                    .append(self.term.doc(alloc))
                    .group()
                    .append(";")
                    .nest(4),
            )
    }
}

impl PartialEq for Alias {
    fn eq(&self, other: &Alias) -> bool {
        self.name == other.name && self.term == other.term
    }
}

/// A struct type definition.
#[derive(Debug, Clone)]
pub struct StructType {
    /// The full span of this definition.
    pub span: Span,
    /// Doc comment.
    pub doc: Arc<[String]>,
    /// Name of this definition.
    pub name: Label,
    /// Fields in the struct.
    pub fields: Vec<TypeField>,
}

impl StructType {
    pub fn doc<'core, D>(&'core self, alloc: &'core D) -> DocBuilder<'core, D>
    where
        D: DocAllocator<'core>,
        D::Doc: Clone,
    {
        let docs = alloc.concat(self.doc.iter().map(|line| {
            (alloc.nil())
                .append("///")
                .append(line)
                .group()
                .append(alloc.newline())
        }));

        let struct_prefix = (alloc.nil())
            .append("struct")
            .append(alloc.space())
            .append(self.name.doc(alloc))
            .append(alloc.space());

        let struct_ty = if self.fields.is_empty() {
            (alloc.nil()).append(struct_prefix).append("{}").group()
        } else {
            (alloc.nil())
                .append(struct_prefix)
                .append("{")
                .group()
                .append(alloc.concat(self.fields.iter().map(|field| {
                    (alloc.nil())
                        .append(alloc.newline())
                        .append(field.doc(alloc))
                        .nest(4)
                        .group()
                })))
                .append(alloc.newline())
                .append("}")
        };

        (alloc.nil()).append(docs).append(struct_ty)
    }
}

impl PartialEq for StructType {
    fn eq(&self, other: &StructType) -> bool {
        self.name == other.name && self.fields == other.fields
    }
}

/// A field in a struct type definition.
#[derive(Debug, Clone)]
pub struct TypeField {
    pub doc: Arc<[String]>,
    pub start: ByteIndex,
    pub name: Label,
    pub term: Term,
}

impl TypeField {
    pub fn span(&self) -> Span {
        Span::new(self.start, self.term.span().end())
    }

    pub fn doc<'core, D>(&'core self, alloc: &'core D) -> DocBuilder<'core, D>
    where
        D: DocAllocator<'core>,
        D::Doc: Clone,
    {
        let docs = alloc.concat(self.doc.iter().map(|line| {
            (alloc.nil())
                .append("///")
                .append(line)
                .group()
                .append(alloc.newline())
        }));

        (alloc.nil())
            .append(docs)
            .append(
                (alloc.nil())
                    .append(self.name.doc(alloc))
                    .append(alloc.space())
                    .append(":")
                    .group(),
            )
            .append(
                (alloc.nil())
                    .append(alloc.space())
                    .append(self.term.doc(alloc))
                    .append(","),
            )
    }
}

impl PartialEq for TypeField {
    fn eq(&self, other: &TypeField) -> bool {
        self.name == other.name && self.term == other.term
    }
}

/// Terms.
#[derive(Debug, Clone, Eq)]
pub enum Term {
    /// Item references
    Item(Span, Label),

    /// Terms annotated with types.
    Ann(Arc<Term>, Arc<Term>),

    /// The sort of kinds.
    Kind(Span),
    /// The kind of types.
    Type(Span),

    /// Unsigned 8-bit integers.
    U8(Span),
    /// Unsigned 16-bit integers (little endian).
    U16Le(Span),
    /// Unsigned 16-bit integers (big endian).
    U16Be(Span),
    /// Unsigned 32-bit integers (little endian).
    U32Le(Span),
    /// Unsigned 32-bit integers (big endian).
    U32Be(Span),
    /// Unsigned 64-bit integers (little endian).
    U64Le(Span),
    /// Unsigned 64-bit integers (big endian).
    U64Be(Span),
    /// Signed, two's complement 8-bit integers.
    S8(Span),
    /// Signed, two's complement 16-bit integers (little endian).
    S16Le(Span),
    /// Signed, two's complement 16-bit integers (big endian).
    S16Be(Span),
    /// Signed, two's complement 32-bit integers (little endian).
    S32Le(Span),
    /// Signed, two's complement 32-bit integers (big endian).
    S32Be(Span),
    /// Signed, two's complement 64-bit integers (little endian).
    S64Le(Span),
    /// Signed, two's complement 64-bit integers (big endian).
    S64Be(Span),
    /// IEEE754 single-precision floating point numbers (little endian).
    F32Le(Span),
    /// IEEE754 single-precision floating point numbers (big endian).
    F32Be(Span),
    /// IEEE754 double-precision floating point numbers (little endian).
    F64Le(Span),
    /// IEEE754 double-precision floating point numbers (big endian).
    F64Be(Span),

    /// Error sentinel.
    Error(Span),
}

impl Term {
    pub fn span(&self) -> Span {
        match self {
            Term::Item(span, _)
            | Term::Kind(span)
            | Term::Type(span)
            | Term::U8(span)
            | Term::U16Le(span)
            | Term::U16Be(span)
            | Term::U32Le(span)
            | Term::U32Be(span)
            | Term::U64Le(span)
            | Term::U64Be(span)
            | Term::S8(span)
            | Term::S16Le(span)
            | Term::S16Be(span)
            | Term::S32Le(span)
            | Term::S32Be(span)
            | Term::S64Le(span)
            | Term::S64Be(span)
            | Term::F32Le(span)
            | Term::F32Be(span)
            | Term::F64Le(span)
            | Term::F64Be(span)
            | Term::Error(span) => *span,
            Term::Ann(term, ty) => Span::merge(term.span(), ty.span()),
        }
    }

    pub fn doc<'core, D>(&'core self, alloc: &'core D) -> DocBuilder<'core, D>
    where
        D: DocAllocator<'core>,
        D::Doc: Clone,
    {
        self.doc_prec(alloc, 0)
    }

    pub fn doc_prec<'core, D>(&'core self, alloc: &'core D, prec: u8) -> DocBuilder<'core, D>
    where
        D: DocAllocator<'core>,
        D::Doc: Clone,
    {
        let show_paren = |cond, doc| match cond {
            true => alloc.text("(").append(doc).append(")"),
            false => doc,
        };

        match self {
            Term::Item(_, label) => (alloc.nil())
                .append("item")
                .append(alloc.space())
                .append(alloc.as_string(label)),
            Term::Ann(term, ty) => show_paren(
                prec > 0,
                (alloc.nil())
                    .append(term.doc_prec(alloc, prec + 1))
                    .append(alloc.space())
                    .append(":")
                    .group()
                    .append(
                        (alloc.space())
                            .append(ty.doc_prec(alloc, prec + 1))
                            .group()
                            .nest(4),
                    ),
            ),
            Term::Kind(_) => alloc.text("Kind"),
            Term::Type(_) => alloc.text("Type"),
            Term::U8(_) => alloc.text("U8"),
            Term::U16Le(_) => alloc.text("U16Le"),
            Term::U16Be(_) => alloc.text("U16Be"),
            Term::U32Le(_) => alloc.text("U32Le"),
            Term::U32Be(_) => alloc.text("U32Be"),
            Term::U64Le(_) => alloc.text("U64Le"),
            Term::U64Be(_) => alloc.text("U64Be"),
            Term::S8(_) => alloc.text("S8"),
            Term::S16Le(_) => alloc.text("S16Le"),
            Term::S16Be(_) => alloc.text("S16Be"),
            Term::S32Le(_) => alloc.text("S32Le"),
            Term::S32Be(_) => alloc.text("S32Be"),
            Term::S64Le(_) => alloc.text("S64Le"),
            Term::S64Be(_) => alloc.text("S64Be"),
            Term::F32Le(_) => alloc.text("F32Le"),
            Term::F32Be(_) => alloc.text("F32Be"),
            Term::F64Le(_) => alloc.text("F64Le"),
            Term::F64Be(_) => alloc.text("F64Be"),
            Term::Error(_) => alloc.text("!"),
        }
    }
}

impl PartialEq for Term {
    fn eq(&self, other: &Term) -> bool {
        match (self, other) {
            (Term::Item(_, label0), Term::Item(_, label1)) => label0 == label1,
            (Term::Ann(term0, ty0), Term::Ann(term1, ty1)) => term0 == term1 && ty0 == ty1,
            (Term::Kind(_), Term::Kind(_))
            | (Term::Type(_), Term::Type(_))
            | (Term::U8(_), Term::U8(_))
            | (Term::U16Le(_), Term::U16Le(_))
            | (Term::U16Be(_), Term::U16Be(_))
            | (Term::U32Le(_), Term::U32Le(_))
            | (Term::U32Be(_), Term::U32Be(_))
            | (Term::U64Le(_), Term::U64Le(_))
            | (Term::U64Be(_), Term::U64Be(_))
            | (Term::S8(_), Term::S8(_))
            | (Term::S16Le(_), Term::S16Le(_))
            | (Term::S16Be(_), Term::S16Be(_))
            | (Term::S32Le(_), Term::S32Le(_))
            | (Term::S32Be(_), Term::S32Be(_))
            | (Term::S64Le(_), Term::S64Le(_))
            | (Term::S64Be(_), Term::S64Be(_))
            | (Term::F32Le(_), Term::F32Le(_))
            | (Term::F32Be(_), Term::F32Be(_))
            | (Term::F64Le(_), Term::F64Le(_))
            | (Term::F64Be(_), Term::F64Be(_))
            | (Term::Error(_), Term::Error(_)) => true,
            (_, _) => false,
        }
    }
}

/// Values.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Item references.
    Item(Label),

    /// The sort of kinds.
    Kind,
    /// The kind of types.
    Type,

    /// Unsigned 8-bit integers.
    U8,
    /// Unsigned 16-bit integers (little endian).
    U16Le,
    /// Unsigned 16-bit integers (big endian).
    U16Be,
    /// Unsigned 32-bit integers (little endian).
    U32Le,
    /// Unsigned 32-bit integers (big endian).
    U32Be,
    /// Unsigned 64-bit integers (little endian).
    U64Le,
    /// Unsigned 64-bit integers (big endian).
    U64Be,
    /// Signed, two's complement 8-bit integers.
    S8,
    /// Signed, two's complement 16-bit integers (little endian).
    S16Le,
    /// Signed, two's complement 16-bit integers (big endian).
    S16Be,
    /// Signed, two's complement 32-bit integers (little endian).
    S32Le,
    /// Signed, two's complement 32-bit integers (big endian).
    S32Be,
    /// Signed, two's complement 64-bit integers (little endian).
    S64Le,
    /// Signed, two's complement 64-bit integers (big endian).
    S64Be,
    /// IEEE754 single-precision floating point numbers (little endian).
    F32Le,
    /// IEEE754 single-precision floating point numbers (big endian).
    F32Be,
    /// IEEE754 double-precision floating point numbers (little endian).
    F64Le,
    /// IEEE754 double-precision floating point numbers (big endian).
    F64Be,

    /// Error sentinel.
    Error,
}
