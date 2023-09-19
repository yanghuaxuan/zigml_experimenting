///! Very basic demonstration of machine learning

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
fn cost(w: f32, b: f32) f32 {
    // Where: y = x * w + b
    // A model of y that predicts the next number given a set of inputs (1st element). The 2nd element is our actual expected y value
    // 
    // We can see that the output is clearly just the input multiplied by 2. 
    // What if we don't know that, or that the actual output is not so easily predictable? 
    //
    // Ignore b (bias) for now, just an extra value that *supposedly* improves the quality of training
    //
    const train = [_][2]f32 {
        [_]f32{0, 0},
        [_]f32{1, 2},
        [_]f32{2, 4},
        [_]f32{2, 4},
        [_]f32{3, 6},
        [_]f32{4, 8},
    };

    var results: f32 = 0.0;

    for (train) |val| {
        const x = val[0];
        const y = x * w + b;
        // Calculate error
        const err = val[1] - y;

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

    var w = rand_float() * 10.0;
    print("W: {}\n", .{w});
    // We want to make this better...
    print("MSE: {}\n", .{cost(w, 0)});

    // Very small number to gently shift the weight to see what would happen...
    var eps: f32 = 1e-3;
    print("MSE + eps: {}\n", .{cost(w + eps, 0)}); // The res may be slightly better, slightly worse...

    // Let's automate this process..
    // We want to minimize the cost (as close to zero as possible)
    // To to that, let's calculate what our new weight will go by.
    var dcost: f32 = (cost(w + eps, 0) - cost(w, 0)) / eps; 

    // Now let's modify the weight by the eps
    w -= dcost;
    print("MSE + eps again: {}\n", .{cost(w, 0)});

    // Maybe it's learning too fast/slow...
    // Let's set a rate!
    var rate: f32 = 1e-3;
    w = rand_float() * 10.0;
    eps = 1e-3;
    var train: usize = 4; // This time we _iterate_ through our cost function 4 more times
    for (0..train) |i| {
        dcost = (cost(w + eps, 0) - cost(w, 0)) / eps;
        w -= dcost * rate;
        // We see that it's actually getting better!
        print("ITER {}: cost = {}, w = {}\n", .{i, cost(w, 0), w});
    }
    
    // Now we have (almost) implemented a single neuron! In order for it to be a real artificial neuron,
    // we must also add a bias variable. Each neuron has its own bias variable, independent of the input.
    var b = rand_float() * 1.0e-15;
    rate = 1e-3;
    w = rand_float() * 10.0;
    eps = 1e-3;
    train = 100; // This time we _iterate_ through our cost function 4 more times
    for (0..train) |i| {
        var dw = (cost(w + eps, b) - cost(w, b)) / eps;
        var db = (cost(w, b + eps) - cost(w, b)) / eps;
        w -= dw * rate;
        b -= db * rate;
        // We see that it's actually getting better!
        print("ITER {}: cost = {}, w = {}, b = {}\n", .{i, cost(w, 0), w, b});
    }
}