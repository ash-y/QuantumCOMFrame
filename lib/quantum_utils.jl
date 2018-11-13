using LinearAlgebra
using TensorOperations
using QuantumOptics
import QuantumOptics:identityoperator
include("../lib/tensor_utils.jl")

function identityoperator(Nfocks::Vector{Int})
    return identityoperator(tensor(FockBasis.(Nfocks)...))
end

function a_vectors(Nfocks::Vector{Int})
    N = length(Nfocks)
    bs = FockBasis.(Nfocks)
    as = Array{Any}(undef, N)
    for (i,b) in enumerate(bs)
        a = identityoperator.(bs)
        a[i] = destroy(b)
        as[i] = reduce(tensor,a)
    end
    return as
end

function ground_state(Nfocks)
    return basisstate(tensor(FockBasis.(Nfocks)...), 1);
end

import Base:*
function *(S::Matrix{<:Number}, z::Vector{SparseOperator})
    return [sum(S[i,:].*z) for i in 1:length(z)]
end


function a2z_mat(N::Int)
    Imat = Matrix{Float64}(I, N, N)
    return [Imat im*Imat ; Imat -im*Imat]/√2
end

function converta2z(Xa::Array{<:Number})
    L = a2z_mat(Int(size(Xa)[1]/2))
    return change_coordinate(Xa,L)
end


#########################################################################
# get μ of arbitrary rank
# ϕ: Ket vector
# z:  a vector of operators (basis)
# rank: the rank of μ tensor we want to get
function get_μ(ϕ::Ket, z::Array{SparseOperator,1}, rank::Int64)
    len = length(z)
    indeces_set = Base.product(ntuple(i->1:len,rank)...)
    size = Base.product(ntuple(i->len,rank)...)
    μ = Array{Float64}(undef,size...)
    for inds in indeces_set
        μ[inds...] = real(expect(reduce(*,(i->z[i]).(collect(inds))),ϕ))
    end
    return symmetrize_tensor(μ)
end



#########################################################################
# get an operator/ operators from a tensor/tensors with a basis z
# X: Tensor
# z: a vector of operators (basis)
function get_operator(X::Array{<:Number},z::Array{SparseOperator,1})
    return sum(X[inds...] * reduce(*,(i->z[i]).(collect(inds))) for inds in Base.product(ntuple(i->1:length(z),ndims(X))...))
end

function get_operator(Xs::Array{Array{Type_Element},1} where Type_Element<:Number,z::Array{SparseOperator,1})
    return sum( get_operator(X,z) for X in Xs)
end

function get_operator(set_of_Xs::Array{Array{Type_Element},2} where Type_Element<:Number,z::Array{SparseOperator,1})
    return [ get_operator(set_of_Xs[i,:],z) for i in 1:size(set_of_Xs)[1]]
end
