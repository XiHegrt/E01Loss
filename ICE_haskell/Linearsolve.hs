module Linearsolve
(vectmult, matvecmult,matmult,linearsolve,triangular,rotatePivot,resubstitute,resubstitute')
where
import Data.List
type Vector = [Double]
type Row    = [Double]
type Matrix = [Row]
vectmult::Vector->Vector->Double
vectmult a b = sum (zipWith (*) a b)

matvecmult :: Matrix -> Vector -> Vector
matvecmult rows v = [sum (zipWith (*) row v) | row <- rows]

matmult :: Matrix -> Matrix -> Matrix
matmult a b
  | null a || null b = []
  | length (head a) /= length b = error "Incompatible matrices for multiplication"
  | otherwise = [[sum (zipWith (*) row col) | col<-(transpose b)] | row<-a]

linearsolve :: Matrix -> Vector -> Vector
linearsolve a b = x
 where
  b' = map (\y -> [y]) b
  a' = zipWith (++) a b'
  x  = resubstitute $ triangular a'


triangular :: Matrix -> Matrix
triangular [] = []
triangular m  = row:(triangular rows')
 where
  (row:rows) = rotatePivot m
  rows' = map f rows
  f bs
   | (head bs) == 0 = drop 1 bs
   | otherwise      = drop 1 $ zipWith (-) (map (*c) bs) row
   where 
    c = (head row)/(head bs)


rotatePivot :: Matrix -> Matrix
rotatePivot (row:rows)
 | (head row) /= 0 = (row:rows)
 | otherwise       = rotatePivot (rows ++ [row])


resubstitute :: Matrix -> Vector
resubstitute = reverse . resubstitute' . reverse . map reverse


resubstitute' :: Matrix -> Vector
resubstitute' [] = []
resubstitute' (row:rows) = x:(resubstitute' rows')
 where
  x     = (head row)/(last row)
  rows' = map substituteUnknown rows
  substituteUnknown (a1:(a2:as')) = ((a1-x*a2):as')


-- Test examples:
-- matA = [[1,-2,6], [4,2,7], [3,11,-5]]
-- vecb = [3,-4,5]
-- x = linearsolve matA vecb
-- matvecmult matA x
