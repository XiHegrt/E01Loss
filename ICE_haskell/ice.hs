module ICE
where

import Linearsolve

-- Input training data
type Vector = [Double]
type Label = Integer
type Item = (Vector, Label)
type Dataset = [Item]
label (x,l) = l
point (x,l) = x


-- Classification model

ones:: Int -> Vector
ones n = take n [1.0,1.0..]

fitw :: Double -> [Vector] -> Vector
fitw sense dx = [-sense] ++ (map (*sense) (linearsolve dx (ones (length (head dx)))))

evalw :: [Vector] -> Vector -> [Double]
evalw dx w = matvecmult (map ([1.0]++) dx) w

plabel :: [Double] -> [Label]
plabel = map (round.signum.underflow)
 where
  smalleps = 1e-8
  underflow v = if (abs v) < smalleps then 0 else v

pclass :: [Vector] -> Vector -> [Label]
pclass dx w = plabel (evalw dx w)


-- Loss

loss01 :: Label -> Label -> Integer
loss01 l1 l2
  | l1 == 0 = 0
  | l2 == 0 = 0
  | l1 /= l2 = 1
  | otherwise = 0

e01 :: [Label] -> [Label] -> Integer
e01 x y = sum (map (\(lx,ly) -> loss01 lx ly) (zip x y))


-- Configuration

type Model = (Vector, Integer)

modelw :: Model -> Vector
modelw (w,l) = w

modell :: Model -> Integer
modell (w,l) = l

-- (data configuration, data sequence, Maybe (prediction model, 0-1 loss))
type Config = ([Item], [Item], Maybe Model) 

comb :: Config -> [Item]
comb (c,s,m) = c

seqn :: Config -> [Item]
seqn (c,s,m) = s

model :: Config -> Maybe Model
model (c,s,m) = m

empty :: Config
empty = ([], [], Nothing)


-- Algorithms

choice fs cs xl = [f' c' xl | f' <- fs, c' <- cs]
choicefilt p fs cs xl = filter p (choice fs cs xl)

e01gen :: Double -> Integer -> Dataset -> [Config]
e01gen sense ub dxl = foldl (choicefilt retain [ignore,include]) [empty] dxl
 where
  dim = length (point (head dxl))
  retain c = (feasible c) && (viable c)
  feasible c = length (comb c) <= dim  
  viable c = case (model c) of
   Nothing -> True
   Just (w,l) -> (l <= ub)

  include c xl = case (model c) of
   Nothing -> if (length updcomb == dim) then
    (updcomb, updseqn, Just (w, e01 (map label updseqn) (pclass (map point updseqn) w)))
   else 
    (updcomb, updseqn, Nothing)
   Just m -> (updcomb, updseqn, Just m)
   where
    updcomb = (comb c) ++ [xl]
    updseqn = (seqn c) ++ [xl]
    w = fitw sense (map point updcomb)

  ignore c xl = case (model c) of
   Nothing -> (comb c, updseqn, Nothing)
   Just (w,l) -> (comb c, updseqn, Just (w, l + e01 [label xl] (pclass [point xl] w)))
   where
    updseqn = (seqn c) ++ [xl]

sel01opt :: [Config] -> Config
sel01opt = foldl1 best
 where
  best c1 c2 = case (model c1) of
   Nothing -> c2
   Just (w1,l1) -> case (model c2) of
    Nothing -> c1
    Just (w2,l2) -> if (l1 <= l2) then c1 else c2

e01class :: Integer -> Dataset -> Config
e01class ub dxy = sel01opt ((e01gen (1.0) ub dxy) ++ (e01gen (-1.0) ub dxy))
