include("../lib/quantum_utils.jl")
using DifferentialEquations
using Random

#######################################################################################
#
# define belavkin equation problem with an ordinary algorithm
#   T: final time
#   ψ0_qo: initial ket
#   A,B,C,D: tensors for Hamiltonian
#   vec_EF: vector of tensors defining Lindblad operators
#
function Belavkin_problem(T::Float64,ψ0::Ket,A::Array{Float64},B1::Array{Float64},B2::Array{Float64},C::Array{Float64},D::Array{Float64},vec_EF::Array{Array{Complex{Float64}},2};seed_input=0)

    Nfocks = ψ0.basis.shape .-1
    N = length(Nfocks)
    as_qo = a_vectors(Nfocks)
    adags_qo = dagger.(as_qo);
    z_qo = vcat((as_qo+adags_qo),(as_qo-adags_qo)/im)/√2;
    zz_qo = [(zi*zj+zj*zi)/2 for zi in z_qo, zj in z_qo];
    Ls = get_operator(vec_EF,z_qo)
    ψ = ground_state(Nfocks)
    dψ = ground_state(Nfocks)

    A′= Array{Complex{Float64}}(undef,2N);
    B′= Array{Complex{Float64}}(undef,2N,2N);
    C′= Array{Complex{Float64}}(undef,2N,2N,2N);
    D′= Array{Complex{Float64}}(undef,2N,2N,2N,2N);


    function f_det(du,u,p,t)
        convert_realvec2complexvec!(ψ.data,u)
        μ₁ = [real(expect(zi_qo,ψ)) for zi_qo in z_qo]
        μ₂ = [real(expect(zzij_qo,ψ)) for zzij_qo in zz_qo]
        B = B1 + B2 * t/T
        get_effective_Hamiltonian_tensors!(A′,B′,C′,D′,A,B,C,D,vec_EF,μ₁,μ₂)
        H = get_operator([A′,B′,C′,D′],z_qo)
        mul!(dψ.data,-im*H.data,ψ.data)
        dif_norm = real(dagger(ψ) * dψ)
        dψ.data -= dif_norm*ψ.data
        convert_complexvec2realvec!(du,dψ.data)
    end

    function f_sto(du,u,p,t)
        convert_realvec2complexvec!(ψ.data,u)
        for i in 1:length(Ls)
            mul!(dψ.data,Ls[i].data,ψ.data)
            dif_norm = real(dagger(ψ) * dψ)
            dψ.data -= dif_norm*ψ.data
            convert_complexvec2realvec!(view(du,:,i),dψ.data)
        end
    end

    u0 = Array{Float64}(undef,2*length(ψ0.data))
    convert_complexvec2realvec!(u0,ψ0.data)

    if seed_input==0
        return SDEProblem(f_det,f_sto,u0,(0.0,T),noise_rate_prototype=zeros(2*length(ψ0.data),size(vec_EF)[1]))
    else
        return SDEProblem(f_det,f_sto,u0,(0.0,T),noise_rate_prototype=zeros(2*length(ψ0.data),size(vec_EF)[1]),seed=seed_input)
    end
end



#######################################################################################
#
# define Schrodinger equation problem with an ordinary algorithm
#   T: final time
#   ψ0_qo: initial ket
#   A,B,C,D: tensors for Hamiltonian
#
#
function Schrodinger_problem(T::Float64,ψ0_qo::Ket,A::Array{Float64},B1::Array{Float64},B2::Array{Float64},C::Array{Float64},D::Array{Float64})

    Nfocks = ψ0_qo.basis.shape .-1
    N = length(Nfocks)
    as_qo = a_vectors(Nfocks)
    adags_qo = dagger.(as_qo);
    z_qo = vcat((as_qo+adags_qo),(as_qo-adags_qo)/im)/√2;
    zz_qo = [(zi*zj+zj*zi)/2 for zi in z_qo, zj in z_qo];

    ψ = Array{Complex{Float64}}(undef,length(ψ0_qo.data))
    dψ = Array{Complex{Float64}}(undef,length(ψ0_qo.data))

    function f_det(du,u,p,t)
        convert_realvec2complexvec!(ψ,u)
        B = B1 + B2 * t/T
        H = get_operator([A,B,C,D],z_qo)
        mul!(dψ,-im*H.data,ψ)
        convert_complexvec2realvec!(du,dψ)
    end

    u0 = Array{Float64}(undef,2*length(ψ0_qo.data))
    convert_complexvec2realvec!(u0,ψ0_qo.data)
    return ODEProblem(f_det,u0,(0.0,T))
end





function get_hist_ϕ(sol::Any,Nfocks::Array{Int64,1},Nsave::Int,T::Float64)
    N = length(Nfocks)

    ϕ_hist = Array{Any}(undef,Nsave)

    for i in 1:Nsave
        ϕ_hist[i] = ground_state(Nfocks)
        state_vec = sol(T*i/Nsave)
        ϕ_hist[i].data[:] .= state_vec[1:length(ϕ_hist[i].data)] + im * state_vec[1+length(ϕ_hist[i].data):end]
    end
    return ϕ_hist
end
