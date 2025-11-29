// stdlib/math.qc
/* 
   Quantica Standard Math Library
*/

/* ============================================
   MATHEMATICAL CONSTANTS
   ============================================ */

let PI = 3.141592653589793
let TAU = 6.283185307179586        /* 2 * PI */
let E = 2.718281828459045           /* Euler's number */
let PHI = 1.618033988749895         /* Golden ratio */
let SQRT2 = 1.4142135623730951      /* Square root of 2 */
let SQRT3 = 1.7320508075688772      /* Square root of 3 */
let SQRT5 = 2.23606797749979        /* Square root of 5 */
let LN2 = 0.6931471805599453        /* Natural log of 2 */
let LN10 = 2.302585092994046        /* Natural log of 10 */
let LOG2E = 1.4426950408889634      /* Log base 2 of E */
let LOG10E = 0.4342944819032518     /* Log base 10 of E */

/* Infinity and NaN representations */
let INF = 1.0 / 0.0
let NEG_INF = -1.0 / 0.0
let NAN = 0.0 / 0.0

/* ============================================
   BASIC ARITHMETIC FUNCTIONS
   ============================================ */

/* Floor function */
func floor(x: Float) -> Float:
    let i = to_int(x)
    let f = to_float(i)
    
    if x < 0.0 And x != f:
        return f - 1.0
    return f

/* Absolute value */
func abs(x: Float) -> Float:
    if x < 0.0:
        return -x
    return x

/* Integer absolute value */
func abs_int(x: Int) -> Int:
    if x < 0:
        return -x
    return x

/* Sign function: returns -1, 0, or 1 */
func sign(x: Float) -> Int:
    if x > 0.0:
        return 1
    elif x < 0.0:
        return -1
    return 0

/* Maximum of two numbers */
func max(a: Float, b: Float) -> Float:
    if a > b:
        return a
    return b

/* Minimum of two numbers */
func min(a: Float, b: Float) -> Float:
    if a < b:
        return a
    return b

/* Maximum of three numbers */
func max3(a: Float, b: Float, c: Float) -> Float:
    return max(max(a, b), c)

/* Minimum of three numbers */
func min3(a: Float, b: Float, c: Float) -> Float:
    return min(min(a, b), c)

/* Clamp value between min and max */
func clamp(x: Float, min_val: Float, max_val: Float) -> Float:
    if x < min_val:
        return min_val
    if x > max_val:
        return max_val
    return x

/* ============================================
   POWER AND EXPONENTIAL FUNCTIONS
   ============================================ */

/* Square root using Newton's method */
func sqrt(x: Float) -> Float:
    if x < 0.0:
        return NAN
    if x == 0.0:
        return 0.0
    
    mut guess = x / 2.0
    mut prev = 0.0
    mut iterations = 0
    
    while abs(guess - prev) > 0.0000001 And iterations < 100:
        prev = guess
        guess = (guess + x / guess) / 2.0
        iterations = iterations + 1
    
    return guess

/* Cube root */
func cbrt(x: Float) -> Float:
    if x == 0.0:
        return 0.0
    
    let sign_x = sign(x)
    let abs_x = abs(x)
    
    mut guess = abs_x / 3.0
    mut prev = 0.0
    mut iterations = 0
    
    while abs(guess - prev) > 0.0000001 And iterations < 100:
        prev = guess
        guess = (2.0 * guess + abs_x / (guess * guess)) / 3.0
        iterations = iterations + 1
    
    if sign_x < 0:
        return -guess
    return guess

/* Exponential function e^x using Taylor series */
func exp(x: Float) -> Float:
    if x > 700.0:
        return INF
    if x < -700.0:
        return 0.0
    
    mut sum = 1.0
    mut term = 1.0
    mut i = 1
    
    while i <= 100:
        term = term * x / i
        sum = sum + term
        if abs(term) < 0.0000000001:
            return sum
        i = i + 1
    
    return sum

/* Natural logarithm using faster converging series */
func ln(x: Float) -> Float:
    if x <= 0.0:
        return NAN
    if x == 1.0:
        return 0.0
    mut val = x
    mut k = 0.0
    while val > 1.5:
        val = val / E
        k = k + 1.0
    
    while val < 0.5:
        val = val * E
        k = k - 1.0
        
    let z = (val - 1.0) / (val + 1.0)
    let z2 = z * z
    
    mut sum = z
    mut term = z
    mut i = 1
    
    
    while i <= 50:
        term = term * z2
        let denom = to_float(2 * i + 1)
        sum = sum + term / denom
        if abs(term) < 1.0e-12:
            
            i = 100
        
        i = i + 1
    
    return k + 2.0 * sum

/* Logarithm base 10 */
func log10(x: Float) -> Float:
    return ln(x) / LN10

/* Logarithm base 2 */
func log2(x: Float) -> Float:
    return ln(x) / LN2

/* Logarithm with arbitrary base */
func log(x: Float, base: Float) -> Float:
    return ln(x) / ln(base)

/* Power function using exp and ln */
func pow(base: Float, exponent: Float) -> Float:
    if base == 0.0:
        if exponent > 0.0:
            return 0.0
        return INF
    
    if base < 0.0 And exponent != floor(exponent):
        return NAN
    
    return exp(exponent * ln(abs(base)))

/* ============================================
   TRIGONOMETRIC FUNCTIONS
   ============================================ */

/* Normalize angle to [-PI, PI] */
func normalize_angle(x: Float) -> Float:
    mut angle = x
    while angle > PI:
        angle = angle - TAU
    while angle < -PI:
        angle = angle + TAU
    return angle

/* Sine using Taylor series */
func sin(x: Float) -> Float:
    let angle = normalize_angle(x)
    mut sum = angle
    mut term = angle
    mut i = 1
    
    while i <= 50:
        term = -term * angle * angle / ((2 * i) * (2 * i + 1))
        sum = sum + term
        if abs(term) < 0.0000000001:
            return sum
        i = i + 1
    
    return sum

/* Cosine using Taylor series */
func cos(x: Float) -> Float:
    let angle = normalize_angle(x)
    mut sum = 1.0
    mut term = 1.0
    mut i = 1
    
    while i <= 50:
        term = -term * angle * angle / ((2 * i - 1) * (2 * i))
        sum = sum + term
        if abs(term) < 0.0000000001:
            return sum
        i = i + 1
    
    return sum

/* Tangent */
func tan(x: Float) -> Float:
    let cos_x = cos(x)
    if abs(cos_x) < 0.0000001:
        return INF
    return sin(x) / cos_x

/* Arcsine using Newton's method */
func asin(x: Float) -> Float:
    if x < -1.0 Or x > 1.0:
        return NAN
    if x == 1.0:
        return PI / 2.0
    if x == -1.0:
        return -PI / 2.0
    
    mut guess = x
    mut i = 0
    
    while i < 20:
        let sin_guess = sin(guess)
        let cos_guess = cos(guess)
        guess = guess - (sin_guess - x) / cos_guess
        i = i + 1
    
    return guess

/* Arccosine */
func acos(x: Float) -> Float:
    if x < -1.0 Or x > 1.0:
        return NAN
    return PI / 2.0 - asin(x)

/* Arctangent using Taylor series with optimization */
func atan(x: Float) -> Float:
    if x < 0.0:
        return -atan(-x)
    if x > 1.0:
        return PI / 2.0 - atan(1.0 / x)
    
    if x > 0.5:
        return PI / 4.0 + atan((x - 1.0) / (1.0 + x))
    
    mut sum = x
    mut term = x
    mut i = 1
    
    while i <= 50:
        term = -term * x * x
        sum = sum + term / to_float(2 * i + 1)
        if abs(term) < 0.0000000001:
            return sum
        i = i + 1
    
    return sum

/* Arctangent of y/x with proper quadrant */
func atan2(y: Float, x: Float) -> Float:
    if x > 0.0:
        return atan(y / x)
    elif x < 0.0 And y >= 0.0:
        return atan(y / x) + PI
    elif x < 0.0 And y < 0.0:
        return atan(y / x) - PI
    elif x == 0.0 And y > 0.0:
        return PI / 2.0
    elif x == 0.0 And y < 0.0:
        return -PI / 2.0
    return 0.0

/* ============================================
   HYPERBOLIC FUNCTIONS
   ============================================ */

/* Hyperbolic sine */
func sinh(x: Float) -> Float:
    let exp_x = exp(x)
    let exp_neg_x = exp(-x)
    return (exp_x - exp_neg_x) / 2.0

/* Hyperbolic cosine */
func cosh(x: Float) -> Float:
    let exp_x = exp(x)
    let exp_neg_x = exp(-x)
    return (exp_x + exp_neg_x) / 2.0

/* Hyperbolic tangent */
func tanh(x: Float) -> Float:
    let exp_2x = exp(2.0 * x)
    return (exp_2x - 1.0) / (exp_2x + 1.0)

/* Inverse hyperbolic sine */
func asinh(x: Float) -> Float:
    return ln(x + sqrt(x * x + 1.0))

/* Inverse hyperbolic cosine */
func acosh(x: Float) -> Float:
    if x < 1.0:
        return NAN
    return ln(x + sqrt(x * x - 1.0))

/* Inverse hyperbolic tangent */
func atanh(x: Float) -> Float:
    if x <= -1.0 Or x >= 1.0:
        return NAN
    return 0.5 * ln((1.0 + x) / (1.0 - x))

/* ============================================
   ROUNDING AND FLOOR/CEILING
   ============================================ */
/* Ceiling function */
func ceil(x: Float) -> Float:
    let i = to_int(x)
    let f = to_float(i)
    if x > 0.0 And x != f:
        return f + 1.0
    return f

/* Round to nearest integer */
func round(x: Float) -> Float:
    return floor(x + 0.5)

/* Truncate towards zero */
func trunc(x: Float) -> Float:
    return to_float(to_int(x))

/* Fractional part */
func frac(x: Float) -> Float:
    return x - trunc(x)

/* ============================================
   ANGULAR CONVERSIONS
   ============================================ */

/* Degrees to radians */
func radians(degrees: Float) -> Float:
    return degrees * PI / 180.0

/* Radians to degrees */
func degrees(radians: Float) -> Float:
    return radians * 180.0 / PI

/* ============================================
   NUMBER THEORY FUNCTIONS
   ============================================ */

/* Greatest Common Divisor */
func gcd(a: Int, b: Int) -> Int:
    mut x = abs_int(a)
    mut y = abs_int(b)
    
    while y != 0:
        let temp = y
        y = x % y
        x = temp
    
    return x

/* Least Common Multiple */
func lcm(a: Int, b: Int) -> Int:
    if a == 0 Or b == 0:
        return 0
    return abs_int(a * b) / gcd(a, b)

/* Factorial */
func factorial(n: Int) -> Int:
    if n < 0:
        return -1
    if n == 0 Or n == 1:
        return 1
    
    mut result = 1
    mut i = 2
    
    while i <= n:
        result = result * i
        i = i + 1
    
    return result

/* Binomial coefficient C(n, k) */
func combination(n: Float, k: Float) -> Float:
    if k < 0.0 Or k > n:
        return 0.0
    if k == 0.0 Or k == n:
        return 1.0
    
    let k_opt = min(k, n - k)
    mut result = 1.0
    
    mut i = 0
    
    
    while to_float(i) < k_opt:
        
        result = result * (n - to_float(i)) / to_float(i + 1)
        i = i + 1
    
    return result

/* Permutation P(n, k) */
func permutation(n: Int, k: Int) -> Int:
    if k < 0 Or k > n:
        return 0
    
    mut result = 1
    mut i = 0
    
    while i < k:
        result = result * (n - i)
        i = i + 1
    
    return result

/* Check if prime */
func is_prime(n: Int) -> Bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 Or n % 3 == 0:
        return False
    
    mut i = 5
    while i * i <= n:
        if n % i == 0 Or n % (i + 2) == 0:
            return False
        i = i + 6
    
    return True

/* ============================================
   SPECIAL MATHEMATICAL FUNCTIONS
   ============================================ */

/* Gamma function approximation (Stirling) */
func gamma(x: Float) -> Float:
    if x <= 0.0:
        return NAN
    
    let sqrt_2pi = sqrt(2.0 * PI)
    return sqrt_2pi * pow(x / E, x) / sqrt(x)

/* Error function approximation */
func erf(x: Float) -> Float:
    let a1 = 0.254829592
    let a2 = -0.284496736
    let a3 = 1.421413741
    let a4 = -1.453152027
    let a5 = 1.061405429
    let p = 0.3275911
    
    let sign_x = sign(x)
    let abs_x = abs(x)
    
    let t = 1.0 / (1.0 + p * abs_x)
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-abs_x * abs_x)
    
    return sign_x * y

/* Complementary error function */
func erfc(x: Float) -> Float:
    return 1.0 - erf(x)

/* ============================================
   DISTANCE AND GEOMETRY
   ============================================ */

/* Euclidean distance in 2D */
func distance_2d(x1: Float, y1: Float, x2: Float, y2: Float) -> Float:
    let dx = x2 - x1
    let dy = y2 - y1
    return sqrt(dx * dx + dy * dy)

/* Euclidean distance in 3D */
func distance_3d(x1: Float, y1: Float, z1: Float, x2: Float, y2: Float, z2: Float) -> Float:
    let dx = x2 - x1
    let dy = y2 - y1
    let dz = z2 - z1
    return sqrt(dx * dx + dy * dy + dz * dz)

/* Manhattan distance */
func manhattan(x1: Float, y1: Float, x2: Float, y2: Float) -> Float:
    return abs(x2 - x1) + abs(y2 - y1)

/* Chebyshev distance */
func chebyshev(x1: Float, y1: Float, x2: Float, y2: Float) -> Float:
    return max(abs(x2 - x1), abs(y2 - y1))

/* Hypotenuse */
func hypot(x: Float, y: Float) -> Float:
    return sqrt(x * x + y * y)

/* ============================================
   STATISTICAL FUNCTIONS
   ============================================ */

/* Mean of array (requires array support) */
func mean(values: Float[], count: Int) -> Float:
    mut sum = 0.0
    mut i = 0
    while i < count:
        sum = sum + values[i]
        i = i + 1
    return sum / count

/* Variance */
func variance(values: Float[], count: Int) -> Float:
    let avg = mean(values, count)
    mut sum_sq = 0.0
    mut i = 0
    
    while i < count:
        let diff = values[i] - avg
        sum_sq = sum_sq + diff * diff
        i = i + 1
    
    return sum_sq / count

/* Standard deviation */
func std_dev(values: Float[], count: Int) -> Float:
    return sqrt(variance(values, count))

/* ============================================
   UTILITY FUNCTIONS
   ============================================ */

/* Linear interpolation */
func lerp(a: Float, b: Float, t: Float) -> Float:
    return a + (b - a) * t

/* Smooth step interpolation */
func smoothstep(edge0: Float, edge1: Float, x: Float) -> Float:
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

/* Map value from one range to another */
func map_range(value: Float, in_min: Float, in_max: Float, out_min: Float, out_max: Float) -> Float:
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

/* Check if number is NaN */
func is_nan(x: Float) -> Bool:
    return x != x

/* Check if number is infinite */
func is_inf(x: Float) -> Bool:
    return x == INF Or x == NEG_INF

/* Check if number is finite */
func is_finite(x: Float) -> Bool:
    return Not is_nan(x) And Not is_inf(x)

/* Sum of integers from 1 to n */
func sum_integers(n: Int) -> Int:
    return n * (n + 1) / 2

/* Sum of squares from 1 to n */
func sum_squares(n: Int) -> Int:
    return n * (n + 1) * (2 * n + 1) / 6

/* Fibonacci number */
func fibonacci(n: Int) -> Int:
    if n <= 1:
        return n
    
    mut a = 0
    mut b = 1
    mut i = 2
    
    while i <= n:
        let temp = a + b
        a = b
        b = temp
        i = i + 1
    
    return b

/* ============================================
   ADVANCED FEATURES (Not in Python)
   ============================================ */

/* Fast inverse square root (Quake III algorithm) */
func fast_inv_sqrt(x: Float) -> Float:
    return 1.0 / sqrt(x)

/* Sigmoid function */
func sigmoid(x: Float) -> Float:
    return 1.0 / (1.0 + exp(-x))

/* Softplus function */
func softplus(x: Float) -> Float:
    return ln(1.0 + exp(x))

/* ReLU (Rectified Linear Unit) */
func relu(x: Float) -> Float:
    if x > 0.0:
        return x
    return 0.0

/* Leaky ReLU */
func leaky_relu(x: Float, alpha: Float) -> Float:
    if x > 0.0:
        return x
    return alpha * x

/* Swish activation */
func swish(x: Float) -> Float:
    return x * sigmoid(x)

/* Gaussian function */
func gaussian(x: Float, mu: Float, sigma: Float) -> Float:
    let exponent = -(x - mu) * (x - mu) / (2.0 * sigma * sigma)
    return exp(exponent) / (sigma * sqrt(TAU))

/* Sinc function */
func sinc(x: Float) -> Float:
    if abs(x) < 0.0000001:
        return 1.0
    return sin(x) / x

/* Beta function */
func beta(x: Float, y: Float) -> Float:
    return gamma(x) * gamma(y) / gamma(x + y)

/* Logistic function */
func logistic(x: Float, L: Float, k: Float, x0: Float) -> Float:
    return L / (1.0 + exp(-k * (x - x0)))
