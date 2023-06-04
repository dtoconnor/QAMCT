"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com

"


module SpinVectorO3Dynamics
    include("..\\utils\\sampling_utils.jl")

    using ..SamplingUtils: hj_info, add_noise, add_cross_talk, discrete_aggregation
    using DifferentialEquations
    using Random
    using LinearAlgebra
    using Interpolations
    using RecursiveArrayTools


    function O3_Bloch_model(theta_vector::Array{Float64, 1},
                             phi_vector::Array{Float64, 1},
                             biases::Array{Float64, 1},
                             JM::Array{Float64, 2},
                             run_time::Float64,
                             open_system::Bool, # stochastic only
                             temp::Float64, # stochastic only
                             damping::Float64, # stochastic only
                             trajectories::Int64, # stochastic only
                             at_fn,
                             bt_fn,
                             ts,
                             abstol::Float64,
                             reltol::Float64;
                             callback=false,
                             callback_tolerance=1e-3)
        nspins = size(theta_vector, 1)
        t_span = (0.0, run_time)
        i1 = collect(1:nspins)
        i2 = collect((nspins + 1):(2*nspins))
        i3 = collect((2*nspins + 1):(3*nspins))
        # convert to the Bloch equations
        Mx = sin.(theta_vector) .* cos.(phi_vector)
        My = sin.(theta_vector) .* sin.(phi_vector)
        Mz = cos.(theta_vector)
        u_0 = vcat(Mx, My, Mz)

        model! = function (du, u, p, t)
            mx, my, mz = u[i1], u[i2], u[i3]
            hx = 2.0 * at_fn(t)
            hz = 2.0 .* bt_fn(t) .* (biases  .+ (JM * mz))
            du[i1] = -hz .* my
            du[i2] = (hx .* mz) + (hz .* mx)
            du[i3] = -hx .* my
            nothing
        end

        noise_model! = function (du, u, p, t)
            mx, my, mz = u[i1], u[i2], u[i3]
            hx = 2.0 * at_fn(t)
            hz = 2.0 .* bt_fn(t) .* (biases  + (JM * mz))
            du[i1] = (-hz .* ((damping .* mx .* mz) + my)) -
                       (damping .* hx .* (mz.^2 + my.^2))

            du[i2] = (hz .* (mx - (damping .* my .* mz))) +
                       (hx .* ((damping .* mx .* my) + mz))

            du[i3] = (-damping .* hz .* (mx.^2 + my.^2)) +
                       (hx .* ((damping .* mx .* mz) - my))
            nothing
        end

        σ_model! = function (du, u, p, t)
            # constants infront of wiener process
            mx, my, mz = u[i1], u[i2], u[i3]
            c = sqrt(2*damping*temp)
            du[i1] = c .* (my + mz)
            du[i2] = c .* (mx + mz)
            du[i3] = c .* (mx + my)
            nothing
        end

        # callback to stipulate a radius of 1

        condition(u,t,integrator) = any((u[i1].^2 + u[i2].^2 + u[i3].^2 .- 1.0).^2 .>=callback_tolerance)
        affect! = function (integrator)
            c = sqrt.((integrator.u[i1].^2 + integrator.u[i2].^2 + integrator.u[i3].^2))
            integrator.u[i1] ./= c
            integrator.u[i2] ./= c
            integrator.u[i3] ./= c
            nothing
        end
        cb = DifferentialEquations.DiscreteCallback(condition, affect!)
        if !callback
            cb = nothing
        end
        # supplying jacobian can provide speed increases

        if open_system
            # f =  DifferentialEquations.ODEFunction(noise_model!, jac=noise_model_jac!)
            f = noise_model!
            prob = DifferentialEquations.SDEProblem(f, σ_model!, u_0, t_span)
            ensembleprob = EnsembleProblem(prob)
            # can be upgraded to use GPU
            sol = DifferentialEquations.solve(ensembleprob, SOSRA(), EnsembleThreads(),
                                              trajectories=trajectories,
                                              saveat=ts, abstol=abstol,
                                              reltol=reltol, maxiters=1e6, callback=cb)
        else
            # f =  DifferentialEquations.ODEFunction(model!, jac=model_jac!)
            f = model!
            prob = DifferentialEquations.ODEProblem(f, u_0, t_span)
            sol = DifferentialEquations.solve(prob, saveat=ts, # callback=cb,
                                              abstol=abstol, reltol=reltol,
                                              maxiters=1e6, callback=cb)
        end

        return sol
    end

    function O3_model(theta_vector::Array{Float64, 1},
                         phi_vector::Array{Float64, 1},
                         biases::Array{Float64, 1},
                         JM::Array{Float64, 2},
                         run_time::Float64,
                         open_system::Bool, # stochastic only
                         temp::Float64, # stochastic only
                         damping::Float64, # stochastic only
                         trajectories::Int64, # stochastic only
                         at_fn,
                         bt_fn,
                         ts,
                         abstol::Float64,
                         reltol::Float64)
        if open_system
            @error ErrorException("Use Bloch equation for open system evaluation")
        end
        nspins = size(theta_vector, 1)
        t_span = (0.0, run_time)
        ω_θ = zeros(Float64, nspins)
        ω_ϕ = zeros(Float64, nspins)
        u_0 = ArrayPartition(theta_vector, ω_θ, phi_vector, ω_ϕ)

        model! = function (du, u, p, t)
            # θ contribution
            θ, ω, ϕ, ρ = u.x[1], u.x[2], u.x[3], u.x[4]
            hx = at_fn(t)
            hz = bt_fn(t) .* (biases + (JM * cos.(θ)))
            du.x[1] .= ω
            du.x[2] .= (-0.5 .* ρ .* sin.(θ)) + ((hx .* cos.(θ) .* cos.(ϕ)) +
                       (sin.(θ) .* hz))

            # ϕ contributions
            du.x[3] .= ρ
            du.x[4] .= (0.5 .* ω .* sin.(θ)) - (hx .* sin.(θ) .* sin.(ϕ))
            nothing
        end

        # f =  DifferentialEquations.ODEFunction(model!, jac=model_jac!)
        f = model!
        prob = DifferentialEquations.ODEProblem(f, u_0, t_span)
        sol = DifferentialEquations.solve(prob, saveat=ts, abstol=abstol,
                                          reltol=reltol, maxiters=1e8)
        return sol
    end



    function solve(h::Dict, J::Dict, run_time=100.0; # in nanoseconds
                    temp=0.1, friction_constant=1e-3, trajectories=1000,
                    # for individual qubit schedules, input a
                    # dictionary of interpolations
                    # NOTE if interpolations aren't continuous then
                    # instabilities arise in the ODE integrator
                    # NOTE Bloch equations are more stable
                    # NOTE opensystem isn't available without bloch equatins
                    a_fn=nothing, b_fn=nothing, Bloch=true,
                    add_xtalk=false, noisy=false, open_system=false,
                    initial_theta_state=Array{Float64, 1}(),
                    initial_phi_state=Array{Float64, 1}(), callback=false,
                    saveat=collect(0:0.01:1),  # points in normalized time
                    abstol=1e-9, reltol=1e-7, kwargs...)

        nspins = length(h)
        # check to see if we have individual qubit schedules
        independent_qubit_sechdules = false
        if isa(a_fn, Dict) | isa(b_fn, Dict)
            # if one input is not a dict but one if, convert to dict of lists
            if !isa(a_fn, Dict)
                interp = deepcopy(a_fn)
                a_fn = Dict([(i, interp) for i = 1:nspins])
            elseif !isa(b_fn, Dict)
                interp = deepcopy(b_fn)
                b_fn = Dict([(i, interp) for i = 1:nspins])
            end

            independent_qubit_sechdules = true
        elseif a_fn == nothing
            # linear schedules
            a_fn = LinearInterpolation(saveat, 1.0 .- saveat, extrapolation_bc=Flat())
        elseif b_fn == nothing
            b_fn = LinearInterpolation(saveat, saveat, extrapolation_bc=Flat())
        end

        # define coefficient interpolation functions
        if independent_qubit_sechdules
            # vectorise the independent_schedules
            at_fn = function (t)
                a_vec = zeros(Float64, nspins)
                for i = 1:nspins
                    a_vec[i] = a_fn[i](t/run_time)
                end
                return a_vec
            end
            bt_fn = function (t)
                b_vec = zeros(Float64, nspins)
                for i = 1:nspins
                    b_vec[i] = b_fn[i](t/run_time)
                end
                return b_vec
            end
        else
            at_fn = (t) -> a_fn(t/run_time)
            bt_fn = (t) -> b_fn(t/run_time)
        end

        # prepare initial state if defined
        if length(initial_theta_state) == nspins
            if length(initial_phi_state) != nspins
                initial_phi_state = zeros(Float64, nspins)
            end
            @assert ndims(initial_theta_state) == 1
            @assert ndims(initial_phi_state) == 1
            theta_vector = deepcopy(initial_theta_state)
            phi_vector = deepcopy(initial_phi_state)
        else
            # starting in the transverse field direction as default
            theta_vector = fill(0.5 * pi, nspins)
            phi_vector = zeros(Float64, nspins)
        end

        # add noise effects
        if noisy
            h, J = add_noise(h, J)
        end
        # add cross-talk effects
        if add_xtalk
            h, J = add_xtalk(h, J)
        end

        # prepare problem info
        JM, M, degs = hj_info(h, J)
        JS = deepcopy(JM)
        biases = diag(JM)
        JM -= diagm(biases)

        if Bloch
            _fn = O3_Bloch_model
        else
            _fn = O3_model
        end

        # run the protocol
        sol = _fn(theta_vector, phi_vector, biases, JM, run_time,
                      open_system, # stochastic only
                      temp, # stochastic only
                      friction_constant, # stochastic only
                      trajectories, # stochastic only
                      at_fn, bt_fn, run_time.*saveat, abstol,
                      reltol, callback=callback)

        z_idxs = (2*nspins + 1):(3*nspins)
        if open_system
            spin_vectors = [Array{Float64, 1}(sign.(sol[i].u[end][z_idxs])) for i =1:trajectories]
            results = discrete_aggregation(spin_vectors, trajectories, JS)
        else
            if !Bloch
                final_sol = sol.u[end].x[1]
                spins = Array{Float64, 1}(sign.(cos.(final_sol)))
            else
                final_sol = sol.u[end][z_idxs]
                spins = Array{Float64, 1}(sign.(final_sol))
            end

            energy = biases' * spins + 0.5 * (spins' * (JM * spins))
            results = [spins, energy]
        end
        return results, sol
    end

    export solve, O3_model, O3_Bloch_model

end
