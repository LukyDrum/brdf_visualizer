pub enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<A, B> Clone for Either<A, B>
where
    A: Clone,
    B: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Left(arg) => Self::Left(arg.clone()),
            Self::Right(arg) => Self::Right(arg.clone()),
        }
    }
}
