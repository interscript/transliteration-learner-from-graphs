
using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Agent.jl")
include("src/Rules.jl")



function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin
        "--path-lucidchart-csv"
            help = "path to the graph csv"
        "--dir-path-lucidchart-csv"
            help = "path to directory containing graph csv's"
        "--brain-entry"
            help = "brain entry"
        "--path-model"
            help = "path to the train model, e.g. resources/model.dat"

    end

    parse_args(s)

end


parsedArgs = parse_commandline()


modelName = parsedArgs["path-model"]
brainEntry = lowercase(parsedArgs["brain-entry"])
entryFound = false

# Parse csv data
df = Nothing
if !isnothing(get(parsedArgs, "dir-path-lucidchart-csv", nothing)) 
    
    println("processing directory::", parsedArgs["dir-path-lucidchart-csv"])
    dirName = parsedArgs["dir-path-lucidchart-csv"]
    df = filter(s -> s[end-3:end] == ".csv", readdir(parsedArgs["dir-path-lucidchart-csv"])) |>
        (gNs -> map(gN -> (println("process file::", gN);
                           DataFrame(CSV.File(dirName*gN))), gNs)) |>
            (vgNs -> vcat(vgNs...))
    
else
    
    println("process file::", parsedArgs["path-lucidchart-csv"])
    df = DataFrame(CSV.File(parsedArgs["path-lucidchart-csv"]))
    
end


# Preprocess Nodes
df[!,"Label"] = map(x -> ismissing(x) ? Missing : lowercase(x), df[!,"Text Area 1"])

df_Nodes = filter(row -> row.Name in ["Decision", "Process", "Terminator"], df)
df_Arrows = filter(row -> row.Name in ["Line"], df);
df_Brains = filter(row -> row.Name in ["Curly Brace Note"], df);


dicBRAINS = Dict{String, Node}()

brainsList = df_Brains[!, "Label"]


if !(brainEntry in brainsList)

    @warn "brain-entry not found in graph!
              (notice that lowercases of node names are taken)"
    exit()

end

for b in brainsList

    @info "build brain: ", b

    try

        dicBRAINS[b] = get_node(b, df_Brains) |>
                (D -> (n=Node(D, nothing); n.x[:depth]=0; n)) |>
                    (N -> createTree(N, df_Nodes, df_Arrows, df_Brains))

    catch

        @error "error! brain : ", b

    end

end


serialize(modelName, Dict(:dicBrains => dicBRAINS,
                          :df_Nodes => df_Nodes,
                          :entry => brainEntry))
println("data saved to: ", modelName)
