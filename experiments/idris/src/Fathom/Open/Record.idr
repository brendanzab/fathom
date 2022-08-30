||| Open format universe
|||
||| This module defines an open universe of binary format descriptions using
||| records to define an interface. By defining formats in this way, the
||| universe of formats is open to extension.
|||
||| I’m not sure, but this reminds me a little of the ‘coinductively defined
||| universes’ that [some type theorists were proposing](https://www.cmu.edu/dietrich/philosophy/hott/slides/shulman-2022-05-12.pdf#page=79),
||| but I may be mistaken.

module Fathom.Open.Record


import Data.Colist
import Data.Vect

import Fathom.Base
import Fathom.Data.Refine
import Fathom.Data.Sing


-------------------------
-- FORMAT DESCRIPTIONS --
-------------------------


public export
record Format where
  constructor MkFormat
  Rep : Type
  decode : Decode (Rep, BitStream) BitStream
  encode : Encode Rep BitStream


--------------
-- FORMATS --
--------------


public export
end : Format
end = MkFormat { Rep, decode, encode } where
  Rep : Type
  Rep = Unit

  decode : Decode (Rep, BitStream) BitStream
  decode [] = Just ((), [])
  decode (_::_) = Nothing

  encode : Encode Rep BitStream
  encode () = Just []


public export
fail : Format
fail = MkFormat { Rep, decode, encode } where
  Rep : Type
  Rep = Void

  decode : Decode (Rep, BitStream) BitStream
  decode _ = Nothing

  encode : Encode Rep BitStream
  encode x = void x


public export
pure : {0 A : Type} -> A -> Format
pure x = MkFormat { Rep, decode, encode } where
  Rep : Type
  Rep = Sing x

  decode : Decode (Rep, BitStream) BitStream
  decode buffer = Just (MkSing x, buffer)

  encode : Encode Rep BitStream
  encode (MkSing _) = Just []


public export
skip : (f : Format) -> (def : f.Rep) -> Format
skip f def = MkFormat { Rep, decode, encode } where
  Rep : Type
  Rep = ()

  decode : Decode (Rep, BitStream) BitStream
  decode buffer = do
    (x, buffer') <- f.decode buffer
    Just ((), buffer')

  encode : Encode Rep BitStream
  encode () = f.encode def


public export
repeat : Nat -> Format -> Format
repeat len f = MkFormat { Rep, decode, encode } where
  Rep : Type
  Rep = Vect len f.Rep

  decode : Decode (Rep, BitStream) BitStream
  decode = go len where
    go : (len : Nat) -> Decode (Vect len f.Rep, BitStream) BitStream
    go 0 buffer = Just ([], buffer)
    go (S len) buffer = do
      (x, buffer') <- f.decode buffer
      (xs, buffer'') <- go len buffer'
      Just (x :: xs, buffer'')

  encode : Encode Rep BitStream
  encode = go len where
    go : (len : Nat) -> Encode (Vect len f.Rep) BitStream
    go 0 [] = Just []
    go (S len) (x :: xs) =
      [| f.encode x <+> go len xs |]


public export
bind : (f : Format) -> (f.Rep -> Format) -> Format
bind f1 f2 = MkFormat { Rep, decode, encode } where
  Rep : Type
  Rep = (x : f1.Rep ** (f2 x).Rep)

  decode : Decode (Rep, BitStream) BitStream
  decode buffer = do
    (x, buffer') <- f1.decode buffer
    (y, buffer'') <- (f2 x).decode buffer'
    Just ((x ** y), buffer'')

  encode : Encode Rep BitStream
  encode (x ** y) =
    [| f1.encode x <+> (f2 x).encode y |]


-- Support for do notation

public export
(>>=) : (f : Format) -> (Rep f -> Format) -> Format
(>>=) = bind



-----------------
-- EXPERIMENTS --
-----------------


||| A format description refined with a fixed representation
public export
FormatOf : (0 Rep : Type) -> Type
FormatOf rep = Refine Format (\f => f.Rep = rep)


toFormatOf : (f : Format) -> FormatOf f.Rep
toFormatOf f = MkRefine f
