
using Graphs
using CSV
using YAML
using DataFrames
using ArgParse
using Serialization
using Logging
#Logging.disable_logging(Logging.Info)


include("src/Graphs.jl")
# include("src/Agent.jl")
include("src/AgentBuildAI.jl")
include("src/UtilsGPT.jl")


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin

        "--path-model"
            help = "path to the trained model"

        "--file-name"
            help = "file-name to be transliterated \n
                    excepted if file-name = data/test.csv \n
                    in that case, tests are performed."

        "--file-name-out"
            help = "file-name for transliterations output."

    end

    parse_args(s)

end


parsedArgs = parse_commandline()


# load brain data
data = deserialize(parsedArgs["path-model"])

entryBrain = data[:entry]
dicBRAINS = data[:dicBrains]
dfNodes = data[:dfNodes]
graph = dicBRAINS[entryBrain]


# prepare data
dataSTATE = Dict{String, Any}(
            "txt" => nothing,
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain)

dicCODE = Dict{String, Functor}()

# dictionary of functions
dicCodeSnippets = Dict{String, Functor}()
println(typeof(dicCodeSnippets))

# read params
dicParams = YAML.load_file("../config/params.yml")
# read the source characters as an array in julia
#CharsFilePath = dicParams["file_paths"]["chars_file_path"]
# read chars array from chars_file_path file, in julia:
#Chars = readlines(CharsFilePath, keep=true)


if parsedArgs["file-name"] in ["data/test_benchmark.csv", "test"] # Run the test

    dfTest = DataFrame(CSV.File("data/test_benchmark_ai.csv"))

    dfTest[!,"translitModel"] =
        map(t ->

                begin

                    state = copy(dataSTATE)
                    #try

                        # computation
                        state["txt"] = chomp(t)
                        buildAgentAI(graph, dicBRAINS, dfNodes, state, dicCodeSnippets)

                    #catch

                    #    println("DBG:: error in: ", state["txt"])

                    #end

                end,

            dfTest[!,"source"])

    #ids = evaluation(dfTest[!, "translit"], dfTest[!, "translitModel"]) #, df_Test[!, "orig"])

    #dfBugs = dfTest[ids,:]

    #println("error summary in: data/test_debug.csv")
    #CSV.write("data/test_debug.csv", dfBugs)


end



