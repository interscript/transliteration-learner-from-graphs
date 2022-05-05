

# Include python library
using PyCall

# load python library
include("libs/code_py.jl")


# void data for computations state
dataM = Dict{String, Any}(
            "state" => nothing,
            "brain" => nothing);


# Dictionary with commands
dicCODE = Dict{String, Functor}()


#####################
# mappings          #
#####################

dicCODE["done, do nothing!"] =
    #===
        Basic form of functors:
            d: data
            e: dicBRAINS
            f: df_Nodes
    ===#
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = d["wrd"]; d), # identity
        Dict(:in => ["wrd"], :out => ["res"]))

dicCODE["has a or b?"] =
    #===
        state is used to express the computation "state"
        here, as shown on the diagram, 2 states: yes and no are expected.
        Solution would work for N arbitrary states.
    ===#
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = occursin("a", d["wrd"]) || occursin("b", d["wrd"]) ?
                        "yes" : "no";
         d), # identity
        Dict(:in => ["wrd"], :out => ["wrd", "state"]))

dicCODE["transliterate a and b"] =
    #===
        Example implementing bash, any external lib could be called that way.
        read in julia allows to run an external code.
    ===#
    Functor((d,e=nothing,f=nothing) ->
        (wrd = d["wrd"];
         d["wrd"] =
            read(pipeline(`echo $wrd`, `sed 's/b/c/ ; 's/a/b/`), String);
         d),
        Dict(:in => ["wrd"], :out => ["wrd"]))

dicCODE["apply mappings to all chars"] =
    #===
        We use a python dictionary to generate the mappings.
        Alternative snippet structure with begin; end;
    ===#
    Functor((d,e=nothing,f=nothing) ->
        begin
            wrd = d["wrd"]
            d["wrd"] = join([py"""d_maps"""[string(c)] for c in wrd
                             if c!='\n'])
            d
        end,
        Dict(:in => ["wrd"], :out => ["wrd"]))


#####################
# preprocessor      #
#####################

dicCODE["normalize the text!"] =
    #===
        Example calling a python function
    ===#
    Functor((d,e=nothing,f=nothing) ->
        (println(d);
         d["txt"] = py"""normalise_filter_txt"""(d["txt"]);
         d),
        Dict(:in => ["txt"], :out => ["txt"]))

dicCODE["tokenize the text!"] =
    #===
        Example running julia code
            Function
                :in => txt
                :out => l_wrds
    ===#
    Functor((d,e=nothing,f=nothing) ->
        (d["l_wrds"] = split(d["txt"]);
         d),
        Dict(:in => ["txt"], :out => ["l_wrds"]))


#####################
# transliteration   #
#####################

dicCODE["apply mappings on each word"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["l_transl_wrds"] =
            map(wrd ->
                begin
                    dd = copy(dataM)
                    dd["wrd"] = wrd
                    # dd["pos"] = d["pos"]
                    interfaceName = "mapping"
                    node = e[interfaceName]
                    runAgent(node, e, f, dd)
                end,
                d["l_wrds"]);
        d),
        Dict(:in => ["l_wrds"], :out => ["l_transl_wrds"]))

dicCODE["join transliterated words"] =
Functor((d,e=nothing,f=nothing) ->
    (d["res"] = join(d["l_transl_wrds"], " ");
     d),
    Dict(:in => ["l_transl_wrds"], :out => ["res"]))
