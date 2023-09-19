const c_std = @cImport(@cInclude("stdlib.h"));
const std = @import("std");
// I have no idea how to get time w/ Zig stdlib
const c_time = @cImport(@cInclude("time.h"));
const random = std.rand.DefaultPrng;
const print = std.debug.print;

fn rand_float() f32 {
    const seed: u64 = @intCast(c_time.time(0));

    var rand = random.init(seed);
    return rand.random().float(f32);
}

// Returns the mean sequare error (the cost) of a model
// Model:
// y' = x1 * w1 + x2 * w2 + b
fn cost(w1: f32, w2: f32, b: f32) f32 {
    const train = [_][3]f32 {
        [3]f32{0, 0, 0},
        [3]f32{0, 1, 1},
        [3]f32{1, 0, 1},
        [3]f32{1, 1, 1},
    };

    var results: f32 = 0.0;

    for (train) |val| {
        const x1 = val[0];
        const x2 = val[1];
        const y = x1 * w1 + x2 * w2 + b;
        // Calculate error
        const err = (val[2]) - y;

        // Calculate square error
        //  On top of making the value always postive, and emphasizes the more
        //  variable data points. Squaring the error also allows for the use 
        //  of gradient (calculus)-based methods for optimization.
        results += err * err;

        // print("TRAIN: actual: {}, expected {}\n", .{y, val[1]});
    }
    // Finally, calculate the mean of the square error
    return results / train.len;
}

pub fn main() !void {
    print("Very rudimentary machine learning\n", .{});
    print("\n", .{});

    var b = rand_float() * 1.0e-15;
    const rate = 1e-3;
    var w1 = rand_float() * 10.0;
    var w2 = rand_float() * 10.0;
    const eps = 1e-3;
    const train = 5000; // This time we _iterate_ through our cost function 4 more times
    for (0..train) |i| {
        var dw1 = (cost(w1 + eps, w2 , b) - cost(w1, w2, b)) / eps;
        var dw2 = (cost(w1, w2 + eps, b) - cost(w1, w2, b)) / eps;
        var db = (cost(w1, w2, b + eps) - cost(w1, w2, b)) / eps;
        w1 -= dw1 * rate;
        w2 -= dw2 * rate;
        b -= db * rate;
        // We see that it's actually getting better!
        print("ITER {}: cost = {}, w1 = {}, w2 = {}, b = {}\n", .{i, cost(w1, w2, b), w1, w2, b});
    }
}