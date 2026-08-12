using BVAR
using Test

@testset "BVAR.jl" begin
    include("test-var-estimation.jl")
    include("test-priors.jl")
end
