include("../lib/tensor_utils.jl")


##############################################################################
# adding Kerr term to B,D tensors
#  B,D : tensors representing an operator (Hamiltonian)
#  K : constant of Keer nonlinearity
#
function Kerr!(B::Array{Complex{Float64},2},D::Array{Complex{Float64},4},Ks::Array{Float64,1})

    N = Int(size(B)[1]*0.5)
    for i in 1:N
        D[i+N,i+N,i,i] += -0.5 * Ks[i]
        B[i+N,i] += 2.0 * 0.5 * Ks[i]
    end
end


##############################################################################
# adding Squeeze term to B tensor
#  B : tensors representing an operator (Hamiltonian)
#  zs : constants of Squeezing strength
#
function Squeeze!(B::Array{Complex{Float64},2},zs::Array{Complex{Float64},1})

    N = Int(size(B)[1]*0.5)
    for i in 1:N
        B[i,i] += 0.5 * im * conj(zs[i])
        B[N+i,N+i] += -0.5 * im * zs[i]
    end
end



##############################################################################
# adding Displacement term to A tensors
#  A: tensor representing an operator
#  αs : constants of Squeezing strength
#
function Displacement!(A::Array{Complex{Float64},1}, αs::Array{Complex{Float64},1})

    N = Int(size(A)[1]*0.5)
    for i in 1:N
        A[i] += -0.5 * im * conj(αs[i])
        A[N+i] += 0.5 * im * αs[i]
    end
end
