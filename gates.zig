const c_std = @cImport(@cInclude("stdlib.h"));
const std = @import("std");
// I have no idea how to get time w/ Zig stdlib
const c_time = @cImport(@cInclude("time.h"));
const random = std.rand.DefaultPrng;
const print = std.debug.print;

const exp = std.math.exp;
const pow = std.math.pow;

fn sigmoid(x: f32) f32 {
    return exp(x) / (exp(x) + 1.0);
}

fn invert_int(v: i32) i32 {
    if (v != 0) return 0;
    return 1;
}

fn round(v: f32) i32 {
    const math = @cImport(@cInclude("math.h"));
    return @as(i32, @intFromFloat(math.roundf(v)));
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

    // OR Sample
    // const sample = [_][3]f32 {
    //     [3]f32{0, 0, 0},
    //     [3]f32{0, 1, 1},
    //     [3]f32{1, 0, 1},
    //     [3]f32{1, 1, 1},
    // };
    // NAND sample
    // const sample = [_][3]f32 {
    //     [3]f32{0, 0, 0},
    //     [3]f32{0, 1, 0},
    //     [3]f32{1, 0, 0},
    //     [3]f32{1, 1, 1},
    // };
    // XOR sample
    // Requires two neurons
    const xor_sample_or = [_][3]f32 {
        [3]f32{0, 0, 0},
        [3]f32{0, 1, 1},
        [3]f32{1, 0, 1},
        [3]f32{1, 1, 1},
    };
    const xor_sample_and = [_][3]f32 {
        [3]f32{0, 0, 0},
        [3]f32{0, 1, 0},
        [3]f32{1, 0, 0},
        [3]f32{1, 1, 1},
    };
    const xor_sample = [_][3]f32 {
        [3]f32{0, 0, 1},
        [3]f32{1, 0, 1},
        [3]f32{0, 1, 1},
        [3]f32{1, 1, 0},
    };

    const train = 1_000_000; // This time we _iterate_ through our cost function 4 more times
    const runtime_zero: usize = 0;

    // Single neuron training
    //
    // var b = rand_float() * 1.0e-15;
    // const rate = 1e-3;
    // var w1 = rand_float() * 10.0;
    // var w2 = rand_float() * 10.0;
    // const eps = 1e-3;
    // var result: f32 = 0;
    // for (0..train) |_| {
    //     var dw1 = (cost(sample[runtime_zero..sample.len], w1 + eps, w2 , b) - cost(sample[runtime_zero..sample.len], w1, w2, b)) / eps;
    //     var dw2 = (cost(sample[runtime_zero..sample.len], w1, w2 + eps, b) - cost(sample[runtime_zero..sample.len], w1, w2, b)) / eps;
    //     var db = (cost(sample[runtime_zero..sample.len], w1, w2, b + eps) - cost(sample[runtime_zero..sample.len], w1, w2, b)) / eps;
    //     w1 -= dw1 * rate;
    //     w2 -= dw2 * rate;
    //     b -= db * rate;
    //     // We see that it's actually getting better!
    //     result = cost(&sample, w1, w2, b);
    //     // print("ITER {}: cost = {}, w1 = {}, w2 = {}, b = {}\n", .{i, result, w1, w2, b});
    // }
    // print("ITER {}: cost = {}, w1 = {}, w2 = {}, b = {}\n", .{train, result, w1, w2, b});

    const eps = 1e-3;
    const rate = 1e-3;
    // Two neuron training (XOR)
    //
    // Get weights for AND
    var a_w1 = rand_float() * 10.0;
    var a_w2 = rand_float() * 10.0;
    var a_b = rand_float() * 1.0e-15;
    for(0..train) |_| {
        var dw1 = (cost(xor_sample_and[runtime_zero..xor_sample_and.len], a_w1 + eps, a_w2 , a_b) - cost(xor_sample_and[runtime_zero..xor_sample_and.len], a_w1, a_w2, a_b)) / eps;
        var dw2 = (cost(xor_sample_and[runtime_zero..xor_sample_and.len], a_w1, a_w2 + eps, a_b) - cost(xor_sample_and[runtime_zero..xor_sample_and.len], a_w1, a_w2, a_b)) / eps;
        var db = (cost(xor_sample_and[runtime_zero..xor_sample_and.len], a_w1, a_w2, a_b + eps) - cost(xor_sample_and[runtime_zero..xor_sample_and.len], a_w1, a_w2, a_b)) / eps;
        a_w1 -= dw1 * rate;
        a_w2 -= dw2 * rate;
        a_b -= db * rate;
    }

    // Get weights for OR
    var o_w1 = rand_float() * 10.0;
    var o_w2 = rand_float() * 10.0;
    var o_b = rand_float() * 1.0e-15;
    for(0..train) |_| {
        var dw1 = (cost(xor_sample_or[runtime_zero..xor_sample_or.len], o_w1 + eps, o_w2 , o_b) - cost(xor_sample_or[runtime_zero..xor_sample_or.len], o_w1, o_w2, o_b)) / eps;
        var dw2 = (cost(xor_sample_or[runtime_zero..xor_sample_or.len], o_w1, o_w2 + eps, o_b) - cost(xor_sample_or[runtime_zero..xor_sample_or.len], o_w1, o_w2, o_b)) / eps;
        var db = (cost(xor_sample_or[runtime_zero..xor_sample_or.len], o_w1, o_w2, o_b + eps) - cost(xor_sample_or[runtime_zero..xor_sample_or.len], o_w1, o_w2, o_b)) / eps;
        o_w1 -= dw1 * rate;
        o_w2 -= dw2 * rate;
        o_b -= db * rate;
    }

    print("Testing model!\n", .{});

    // Single neuron sample
    // Should be very close to its expected value!
    // for (sample) |row| {
    //     print("x1: {}, x2: {}, y: {}\n", .{@as(i32, @intFromFloat(row[0])), @as(i32, @intFromFloat(row[1])), sigmoid(row[0] * w1 + row[1] * w2 + b)});
    // }
    
    // Two neuron sample
    // Should be very close to its expected value!
    for (xor_sample) |row| {
        // (x1 | x2) & ~(x1 & x2)
        const x1x2or = sigmoid(row[0] * o_w1 + row[1] * o_w2 + o_b);
        const x1x2iand = @as(f32, @floatFromInt(invert_int(round(sigmoid(row[0] * a_w1 + row[1] * a_w2 + a_b)))));
        print("x1: {}, x2: {}, y: {}\n", .{@as(i32, @intFromFloat(row[0])), 
                                           @as(i32, @intFromFloat(row[1])), 
                                           sigmoid(x1x2or * a_w1 + x1x2iand * a_w2 + a_b)});
    }
}