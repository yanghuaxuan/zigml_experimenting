const c_std = @cImport(@cInclude("stdlib.h"));
const std = @import("std");
// I have no idea how to get time w/ Zig stdlib
const c_time = @cImport(@cInclude("time.h"));
const random = std.rand.DefaultPrng;
const print = std.debug.print;

const exp = std.math.exp;
const pow = std.math.pow;

fn sigmoid(x: f64) f64 {
    return exp(x) / (exp(x) + 1.0);
}

fn rand_float() f32 {
    const seed: u64 = @intCast(c_time.time(0));

    var rand = random.init(seed);
    return rand.random().float(f32);
}

// Returns the mean sequare error (the cost) of a model with 2 inputs (x1, x2), and one expected output (y)
fn cost(sample: []const [3]f32, w1: f32, w2: f32, b: f32) f32 {
    var results: f32 = 0.0;

    for (sample) |val| {
        const x1 = val[0];
        const x2 = val[1];
        const y: f32 = @floatCast(sigmoid(x1 * w1 + x2 * w2 + b));
        // Calculate error
        const err = y - (val[2]);

        // Calculate square error
        //  On top of making the value always postive, and emphasizes the more
        //  variable data points. Squaring the error also allows for the use 
        //  of gradient (calculus)-based methods for optimization.
        results += err * err;

        // print("TRAIN: actual: {}, expected {}\n", .{y, val[1]});
    }
    // Finally, calculate the mean of the square error
    const len: f32 = @floatFromInt(sample.len);
    return results / len;
}

pub fn main() !void {
    print("Very rudimentary machine learning\n", .{});
    print("\n", .{});

    const sample_or = [_][3]f32 {
        [3]f32{0, 0, 0},
        [3]f32{0, 1, 1},
        [3]f32{1, 0, 1},
        [3]f32{1, 1, 1},
    };

    var b = rand_float() * 1.0e-15;
    const rate = 1e-3;
    var w1 = rand_float() * 10.0;
    var w2 = rand_float() * 10.0;
    const eps = 1e-3;
    const train = 100000; // This time we _iterate_ through our cost function 4 more times
    var result: f32 = 0;
    var runtime_zero: usize = 0;

    for (0..train) |_| {
        var dw1 = (cost(sample_or[runtime_zero..sample_or.len], w1 + eps, w2 , b) - cost(sample_or[runtime_zero..sample_or.len], w1, w2, b)) / eps;
        var dw2 = (cost(sample_or[runtime_zero..sample_or.len], w1, w2 + eps, b) - cost(sample_or[runtime_zero..sample_or.len], w1, w2, b)) / eps;
        var db = (cost(sample_or[runtime_zero..sample_or.len], w1, w2, b + eps) - cost(sample_or[runtime_zero..sample_or.len], w1, w2, b)) / eps;
        w1 -= dw1 * rate;
        w2 -= dw2 * rate;
        b -= db * rate;
        // We see that it's actually getting better!
        result = cost(&sample_or, w1, w2, b);
        // print("ITER {}: cost = {}, w1 = {}, w2 = {}, b = {}\n", .{i, result, w1, w2, b});
    }
    // print("ITER {}: cost = {}, w1 = {}, w2 = {}, b = {}\n", .{train, result, w1, w2, b});

    print("Testing model!\n", .{});

    
    // Should be very close to its expected value!
    for (sample_or) |row| {
        print("x1: {}, x2: {}, y: {}\n", .{@as(i32, @intFromFloat(row[0])), @as(i32, @intFromFloat(row[1])), sigmoid(row[0] * w1 + row[1] * w2 + b)});
    }
}