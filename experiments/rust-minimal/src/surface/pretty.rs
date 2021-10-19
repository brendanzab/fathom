use pretty::{Doc, DocAllocator, DocBuilder, DocPtr, RefDoc};
use scoped_arena::Scope;

use crate::surface::Term;
use crate::{StringId, StringInterner};

/// Term precedences
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Prec {
    Top = 0,
    Let,
    Fun,
    App,
    Atomic,
}

pub struct Context<'doc> {
    interner: &'doc StringInterner,
    scope: &'doc Scope<'doc>,
}

impl<'doc> Context<'doc> {
    pub fn new(interner: &'doc StringInterner, scope: &'doc Scope<'doc>) -> Context<'doc> {
        Context { interner, scope }
    }

    pub fn string_id(&'doc self, name: StringId) -> DocBuilder<'doc, Self> {
        self.text(self.interner.resolve(name).unwrap_or("<ERROR>"))
    }

    pub fn ann<Range>(
        &'doc self,
        expr: &Term<'_, Range>,
        r#type: &Term<'_, Range>,
    ) -> DocBuilder<'doc, Self> {
        self.concat([
            self.concat([
                self.term_prec(Prec::Let, &expr),
                self.space(),
                self.text(":"),
            ])
            .group(),
            self.softline(),
            self.term_prec(Prec::Top, &r#type),
        ])
    }

    pub fn paren(&'doc self, wrap: bool, doc: DocBuilder<'doc, Self>) -> DocBuilder<'doc, Self> {
        if wrap {
            self.concat([self.text("("), doc, self.text(")")])
        } else {
            doc
        }
    }

    pub fn term<Range>(&'doc self, term: &Term<'_, Range>) -> DocBuilder<'doc, Self> {
        self.term_prec(Prec::Top, term)
    }

    pub fn term_prec<Range>(
        &'doc self,
        prec: Prec,
        term: &Term<'_, Range>,
    ) -> DocBuilder<'doc, Self> {
        // FIXME: indentation and grouping

        match term {
            Term::Name(_, name) => self.string_id(*name),
            Term::Hole(_, None) => self.text("_"),
            Term::Hole(_, Some(name)) => self.concat([self.text("_"), self.string_id(*name)]),
            Term::Ann(_, expr, r#type) => self.ann(expr, r#type),
            Term::Let(_, (_, def_name), def_type, def_expr, output_expr) => self.paren(
                prec > Prec::Let,
                self.concat([
                    self.concat([
                        self.text("let"),
                        self.space(),
                        self.string_id(*def_name),
                        self.space(),
                        match def_type {
                            None => self.nil(),
                            Some(def_type) => self.concat([
                                self.text(":"),
                                self.softline(),
                                self.term_prec(Prec::Fun, def_type),
                                self.space(),
                            ]),
                        },
                        self.text("="),
                        self.softline(),
                        self.term_prec(Prec::Let, def_expr),
                        self.text(";"),
                    ])
                    .group(),
                    self.line(),
                    self.term_prec(Prec::Let, output_expr),
                ]),
            ),
            Term::Universe(_) => self.text("Type"),
            Term::FunType(_, (_, input_name), input_type, output_type) => self.paren(
                prec > Prec::Fun,
                self.concat([
                    self.concat([
                        self.text("fun"),
                        self.space(),
                        self.text("("),
                        self.string_id(*input_name),
                        self.space(),
                        self.text(":"),
                        self.softline(),
                        self.term_prec(Prec::Top, input_type),
                        self.text(")"),
                        self.space(),
                        self.text("->"),
                    ])
                    .group(),
                    self.softline(),
                    self.term_prec(Prec::Fun, output_type),
                ]),
            ),
            Term::FunArrow(_, input_type, output_type) => self.paren(
                prec > Prec::Fun,
                self.concat([
                    self.term_prec(Prec::App, input_type),
                    self.softline(),
                    self.text("->"),
                    self.softline(),
                    self.term_prec(Prec::Fun, output_type),
                ]),
            ),
            Term::FunIntro(_, (_, input_name), output_expr) => self.paren(
                prec > Prec::Fun,
                self.concat([
                    self.concat([
                        self.text("fun"),
                        self.space(),
                        self.string_id(*input_name),
                        self.space(),
                        self.text("=>"),
                    ])
                    .group(),
                    self.space(),
                    self.term_prec(Prec::Fun, output_expr),
                ]),
            ),
            Term::FunElim(_, head_expr, input_expr) => self.paren(
                prec > Prec::App,
                self.concat([
                    self.term_prec(Prec::App, head_expr),
                    self.space(),
                    self.term_prec(Prec::Atomic, input_expr),
                ]),
            ),
            Term::RecordType(_, type_fields) => self.concat([
                self.text("{"),
                self.space(),
                self.intersperse(
                    type_fields.iter().map(|((_, label), r#type)| {
                        self.concat([
                            self.string_id(*label),
                            self.space(),
                            self.text(":"),
                            self.space(),
                            self.term_prec(Prec::Top, r#type),
                        ])
                    }),
                    self.concat([self.text(","), self.space()]),
                ),
                self.space(),
                self.text("}"),
            ]),
            Term::RecordIntro(_, expr_fields) => self.concat([
                self.text("{"),
                self.space(),
                self.intersperse(
                    expr_fields.iter().map(|((_, label), r#expr)| {
                        self.concat([
                            self.string_id(*label),
                            self.space(),
                            self.text("="),
                            self.space(),
                            self.term_prec(Prec::Top, r#expr),
                        ])
                    }),
                    self.concat([self.text(","), self.space()]),
                ),
                self.space(),
                self.text("}"),
            ]),
            Term::RecordEmpty(_) => self.text("{}"),
            Term::RecordElim(_, head_expr, (_, label)) => self.concat([
                self.term_prec(Prec::Atomic, head_expr),
                self.text("."),
                self.string_id(*label),
            ]),

            Term::U8Type(_) => self.text("U8"),
            Term::U16Type(_) => self.text("U16"),
            Term::U32Type(_) => self.text("U32"),
            Term::U64Type(_) => self.text("U64"),
            Term::S8Type(_) => self.text("S8"),
            Term::S16Type(_) => self.text("S16"),
            Term::S32Type(_) => self.text("S32"),
            Term::S64Type(_) => self.text("S64"),
            Term::F32Type(_) => self.text("F32"),
            Term::F64Type(_) => self.text("F64"),
            Term::NumberLiteral(_, number) => self.string_id(*number),

            Term::FormatType(_) => self.text("Format"),
            Term::FormatRecord(_, format_fields) => self.concat([
                self.text("{"),
                self.space(),
                self.intersperse(
                    format_fields.iter().map(|((_, label), format)| {
                        self.concat([
                            self.string_id(*label),
                            self.space(),
                            self.text("<-"),
                            self.space(),
                            self.term_prec(Prec::Top, format),
                        ])
                    }),
                    self.concat([self.text(","), self.space()]),
                ),
                self.space(),
                self.text("}"),
            ]),
            Term::FormatFail(_) => self.text("fail"),
            Term::FormatU8(_) => self.text("u8"),
            Term::FormatU16Be(_) => self.text("u16be"),
            Term::FormatU16Le(_) => self.text("u16le"),
            Term::FormatU32Be(_) => self.text("u32be"),
            Term::FormatU32Le(_) => self.text("u32le"),
            Term::FormatU64Be(_) => self.text("u64be"),
            Term::FormatU64Le(_) => self.text("u64le"),
            Term::FormatS8(_) => self.text("s8"),
            Term::FormatS16Be(_) => self.text("s16be"),
            Term::FormatS16Le(_) => self.text("s16le"),
            Term::FormatS32Be(_) => self.text("s32be"),
            Term::FormatS32Le(_) => self.text("s32le"),
            Term::FormatS64Be(_) => self.text("s64be"),
            Term::FormatS64Le(_) => self.text("s64le"),
            Term::FormatF32Be(_) => self.text("f32be"),
            Term::FormatF32Le(_) => self.text("f32le"),
            Term::FormatF64Be(_) => self.text("f64be"),
            Term::FormatF64Le(_) => self.text("f64le"),
            Term::FormatRepr(_, expr) => self.paren(
                prec > Prec::App,
                self.concat([
                    self.text("Repr"),
                    self.space(),
                    self.term_prec(Prec::Atomic, expr),
                ]),
            ),

            Term::ReportedError(_) => self.text("_"),
        }
    }
}

// NOTE: based on the `DocAllocator` implementation for `pretty::Arena`
impl<'doc, A: 'doc> DocAllocator<'doc, A> for Context<'doc> {
    type Doc = RefDoc<'doc, A>;

    #[inline]
    fn alloc(&'doc self, doc: Doc<'doc, Self::Doc, A>) -> Self::Doc {
        RefDoc(match doc {
            // Return 'static references for common variants to avoid some allocations
            Doc::Nil => &Doc::Nil,
            Doc::Line => &Doc::Line,
            Doc::Fail => &Doc::Fail,
            // space()
            Doc::BorrowedText(" ") => &Doc::BorrowedText(" "),
            // line()
            Doc::FlatAlt(RefDoc(Doc::Line), RefDoc(Doc::BorrowedText(" "))) => {
                &Doc::FlatAlt(RefDoc(&Doc::Line), RefDoc(&Doc::BorrowedText(" ")))
            }
            // line_()
            Doc::FlatAlt(RefDoc(Doc::Line), RefDoc(Doc::Nil)) => {
                &Doc::FlatAlt(RefDoc(&Doc::Line), RefDoc(&Doc::Nil))
            }
            // softline()
            Doc::Group(RefDoc(Doc::FlatAlt(RefDoc(Doc::Line), RefDoc(Doc::BorrowedText(" "))))) => {
                &Doc::Group(RefDoc(&Doc::FlatAlt(
                    RefDoc(&Doc::Line),
                    RefDoc(&Doc::BorrowedText(" ")),
                )))
            }
            // softline_()
            Doc::Group(RefDoc(Doc::FlatAlt(RefDoc(Doc::Line), RefDoc(Doc::Nil)))) => {
                &Doc::Group(RefDoc(&Doc::FlatAlt(RefDoc(&Doc::Line), RefDoc(&Doc::Nil))))
            }
            _ => self.scope.to_scope(doc),
        })
    }

    fn alloc_column_fn(
        &'doc self,
        f: impl 'doc + Fn(usize) -> Self::Doc,
    ) -> <Self::Doc as DocPtr<'doc, A>>::ColumnFn {
        self.scope.to_scope(f)
    }

    fn alloc_width_fn(
        &'doc self,
        f: impl 'doc + Fn(isize) -> Self::Doc,
    ) -> <Self::Doc as DocPtr<'doc, A>>::WidthFn {
        self.scope.to_scope(f)
    }
}
