use crate::{core, stratified};

struct Context {}

enum EitherLevel<Term, Type, Kind> {
    Term(Term),
    Type(Type),
    Kind(Kind),
}

fn compile_module(core_module: &core::Module) -> stratified::Module {
    unimplemented!()
}

fn compile_item(context: &Context, core_item: &core::Item) -> stratified::Item {
    unimplemented!()
}

fn compile_term(
    context: &Context,
    core_term: &core::Term,
) -> EitherLevel<stratified::Term, stratified::Type, ()> {
    unimplemented!()
}
