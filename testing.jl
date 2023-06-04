"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com

File used to profile and benchmark the algorithms implmented.
"

include("samplers/tools.jl")
include("samplers/SA.jl")
include("samplers/SVMC.jl")
include("samplers/SVMC_TF.jl")
include("samplers/pimc.jl")
using Random
using ..SA
using ..SVMC
using ..SVMCTF
using ..PIMC
using ..SamplingTools
using LinearAlgebra
using SharedArrays
using BenchmarkTools
using NPZ
using Interpolations

function main(test=3)
    # test 0 =  SA, test 1 = SVMC, test 2 = SVMCTF, test 3 = PIMC

    # simple PFC problem
    d = 0.1
    h = Dict([(1, -1.0), (2, 1.0 - d), (3, 1.0-d), (4, -1.0)])
    J = Dict([((1, 2), -1.0), ((2, 3), -1.0), ((3, 4), -1.0)])

    num_samples = 1000
    num_sweeps = 10000
    num_slices = 20
    add_xtalk = false
    add_noise = false

    temp = 12.26 * 1.380649e-23 / (1e12 * 6.62607015e-34)

    a_sch = 1.0 .- collect(LinRange(0, 1, num_sweeps))
    b_sch = collect(LinRange(0, 1, num_sweeps))

    # schedule_data = npzread("data\\DWAVE_schedule.npz")
    # a_fn = LinearInterpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 2])
    # b_fn = LinearInterpolation(schedule_data[:, 1], 0.5.*schedule_data[:, 3])
    # a_sch = a_fn.(collect(LinRange(0, 1, num_sweeps)))
    # b_sch = b_fn.(collect(LinRange(0, 1, num_sweeps)))

    # a_sch_dict = Dict([(1, deepcopy(a_sch)), (2, 0.0 .* deepcopy(a_sch)),
    #                    (3, deepcopy(a_sch)), (4, deepcopy(a_sch)),])
    # b_sch_dict = Dict([(1, deepcopy(b_sch)), (2, deepcopy(b_sch)),
    #                    (3, deepcopy(b_sch)), (4, deepcopy(b_sch)),])

    if test == 0
        result = SA.sample(h, J, num_reads=num_samples, 
                             add_xtalk=add_xtalk, noisy=add_noise)
    elseif test == 1
        result = SVMC.sample(h, J, num_reads=num_samples, temp=temp,
                             a_sch=a_sch, b_sch=b_sch;
                             add_xtalk=add_xtalk, noisy=add_noise)
    elseif test == 2
        result = SVMCTF.sample(h, J, num_reads=num_samples, temp=temp,
                              a_sch=a_sch, b_sch=b_sch;
                                add_xtalk=add_xtalk, noisy=add_noise)
    else

        result = PIMC.sample(h, J, num_reads=num_samples, temp=temp,
                             a_sch=a_sch, b_sch=b_sch, nslices=num_slices;
                             add_xtalk=add_xtalk, noisy=add_noise)
    end

   print("Solutions = $(result[1]), energies = $(result[2]), counts = $(result[3])")

   
end
