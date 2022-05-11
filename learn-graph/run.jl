

using Graphs
using CSV
using JSON
using YAML
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Rules.jl")
include("src/Agent.jl")
include("src/Metrics.jl")

using Logging
Logging.disable_logging(Logging.Info)

using PyCall

hazm = pyimport("hazm")

PATH_HAZM = "resources/postagger.model"
PATH_FLEXICON = "resources/"


stemmer = hazm.Stemmer()
lemmatizer = hazm.Lemmatizer()
normalizer = hazm.Normalizer()
tagger = hazm.POSTagger(model=PATH_HAZM)


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
df_Nodes = data[:df_Nodes]
graph = dicBRAINS[entryBrain]


# prepare data
dataM = Dict{String, Any}(
            "word" => nothing,
            "pos" => nothing,
            "pre_pos" => nothing,
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain) # current brain or graph


dicParams = YAML.load_file("../config/params.yml")
SOURCECHARS = dicParams["transliteration"]["SOURCECHARS"]
#SOURCECHARS =" ءآأؤئابةتثجحخدذرزسشصضطظعغـفقلمنهوَِْپچژکگی \u200c "


function processPOS(pos)

    l_supported_POS = vcat(py"""l_PoS""",
                       collect(keys(py"""d_map_FLEXI""")),
                       collect(keys(py"""d_map_HAZM""")))

    if !(pos in l_supported_POS)

        @error pos, "pos unrecognised, needs to be within: ", l_supported_POS
        exit()

    else

        if pos in collect(keys(py"""d_map_FLEXI"""))

            pos = py"""d_map_FLEXI"""[pos]

        elseif pos in collect(keys(py"""d_map_HAZM"""))

            pos = py"""d_map_HAZM"""[pos]

        end

    end

    pos

end


dataSTATE = Dict{String, Any}(
            "text" => nothing, #parsedArgs["text"],
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain)

if parsedArgs["file-name"] in ["data/test.csv", "test"] # Run the test


    df_Test = DataFrame(CSV.File("data/test.csv"))

    df_Test[!,"transModel"] =
        map(d ->
            begin

                dd = copy(dataSTATE)  
                
                dd["text"] = join(filter(c -> c in SOURCECHARS, d), "") # d

                try

                    wrd = runAgent(graph, dicBRAINS, df_Nodes, dd) |>
                        (w -> replace(w, "-''"=>"", "-'"=>""))
                    wrd

                catch

                    println("DBG:: ", dd["text"]) #, " : ", dd["pos"]);
                    join(dd["text"], " ")

                 end
            end,
            df_Test[!,"orig"])

    ids = evaluation(df_Test[!, "trans"], df_Test[!, "transModel"])
    df_Bugs = df_Test[ids,:]
    select!(df_Bugs, ["Unnamed: 0", "label", "orig", "trans", "transModel"])
    
    println("error summary in: data/test_debug.csv")
    CSV.write("data/test_debug.csv", df_Bugs)


else # transliterate the file

    fileNameOUT = parsedArgs["file-name-out"]
    fileNameOUT = isnothing(fileNameOUT) ?
                        nothing : open(parsedArgs["file-name-out"], "w")
    global fileNameOUT

    using ProgressBars

    doPreprocessData = dicParams["transliteration"]["preprocessData"]
    window = dicParams["transliteration"]["window"]
    space = dicParams["transliteration"]["space"]

    function preprocessData(data)

        if doPreprocessData
            d_data = []
            for d in ProgressBar(data)

                w = split(d)
                for i=1:space:length(w)

                    push!(d_data,
                        w[i:min(end,i+window)])

                    if i+window > length(w)
                        break
                    end

                end
            end
            map(d -> join(d, " "), d_data)
        else
            data
        end

    end

    ProgressBar(readlines(parsedArgs["file-name"], keep=true) |>
                            preprocessData |>
                                (D -> filter(s -> strip(s) != "", D))) |>
        (D ->
            map(t ->
                begin

                    state = copy(dataSTATE)
                    txt = filter(s -> s in SOURCECHARS, chomp(t))
                    state["text"] = txt

                    try

                        # computation
                        translit = runAgent(graph, dicBRAINS, df_Nodes, state) |>
                            (w -> replace(w, "-''"=>"", "-'"=>""))
                        #translit = runAgent(graph, dicBRAINS, df_Nodes, state)
                        global translit
                        # write line by line to STDOUT or file
                        isnothing(fileNameOUT) ?
                            println(string(txt, "|",translit)) :
                            CSV.write(fileNameOUT,
                                    DataFrame(Dict("source" => txt,
                                                   "translit" => translit)),
                                              append=true)

                    catch

                        println("DBG:: error in: ", txt)

                    end

            end,
            D))

end
