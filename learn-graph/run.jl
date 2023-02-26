

using Graphs
using CSV
using YAML
using DataFrames
using ArgParse
using Serialization
using Logging
Logging.disable_logging(Logging.Info)
# using PyCall


include("src/Graphs.jl")
include("src/Agent.jl")
include("src/Metrics.jl")
include("src/Rules.jl")


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin

        "--path-model"
            help = "path to the train model"

        "--file-name"
            help = "file-name to be transliterated \n
                    excepted if file-name = data/test.csv \n
                    in that case, tests are performed."

        "--file-name-out"
            help = "file-name for transliterations output."

    end

    parse_args(s)

end


# parse commands
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


dicParams = YAML.load_file("../config/params.yml")
# read the source characters as an array in julia

CharsFilePath = dicParams["file_paths"]["chars_file_path"]
# read chars array from chars_file_path file, in julia:
Chars = readlines(CharsFilePath, keep=true)


# Run the test
if parsedArgs["file-name"] in ["data/test_benchmark.csv", "test"] 

    dfTest = DataFrame(CSV.File("data/test_benchmark.csv"))

    dfTest[!,"translitModel"] =
        map(t ->

                begin

                    state = copy(dataSTATE)
                    try

                        # computation
                        state["txt"] = chomp(t)
                        runAgent(graph, dicBRAINS, dfNodes, state)

                    catch

                        println("DBG:: error in: ", state["txt"])

                    end

                end,

            dfTest[!,"source"])

    ids = evaluation(dfTest[!, "translit"], dfTest[!, "translitModel"]) #, df_Test[!, "orig"])

    dfBugs = dfTest[ids,:]

    println("error summary in: data/test_debug.csv")
    CSV.write("data/test_debug.csv", dfBugs)


else # transliterate the file

    fileNameOUT = parsedArgs["file-name-out"]

    fileNameOUT = isnothing(fileNameOUT) ?
                        nothing : open(parsedArgs["file-name-out"], "w")

    using ProgressBars

    doPreprocessData = dicParams["transliteration"]["preprocessData"]
    window = dicParams["transliteration"]["window"]
    space = dicParams["transliteration"]["space"]

    function preprocessData(data)

        #===
            function preprocessing data,
            it is useful to shorten sentences so that data can be
            used to train nnetworks.

            window: max length of the text snippets
            space: shift for building snippets
        ===#

        d_data = []
        for d in ProgressBar(data)

            w = split(d)
            for i=1:space:length(w)

                push!(d_data,
                      join(w[i:min(end,i+window)], " "))

                if i+window > length(w)
                    break
                end

            end
        end
        d_data

    end

    ProgressBar(readlines(parsedArgs["file-name"], keep=true) |>
                        (D ->
                            filter(s -> strip(s) != "", D))) |>
                            (D ->
                                doPreprocessData ? preprocessData(D) : D) |>

      (D ->
        map(t ->
                begin

                    state = copy(dataSTATE)

                    try

                        # computation
                        txt = chomp(t)
                        state["txt"] = txt
                        translit = runAgent(graph, dicBRAINS, dfNodes, state)

                        # write line by line to STDOUT or file
                        isnothing(fileNameOUT) ?
                            println(string(txt, "|",translit)) :
                            CSV.write(fileNameOUT,
                                      DataFrame(Dict("source" => txt,
                                                     "translit" => translit)),
                                      append=true)

                    catch

                        println("DBG:: error in: ", state["txt"])

                    end

                end,
            D))

end
