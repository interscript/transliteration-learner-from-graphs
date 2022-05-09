
# https://lucid.app/lucidchart/008ff36b-136f-4666-b06e-ac3b2a8e8d85/edit?page=0_0&invitationId=inv_0221b3ba-bd16-4499-8f60-b95aa68af45c

# Read Mappings
dISOMapping = map(x -> split(x), #|> println,
    readlines("resources/Strict+ShortVowelsOnlyIf.txt")) |>
    (D -> filter(x -> length(x) > 1, D)) |>
    (D -> map(d -> d[1] => d[2], D)) |> Dict ;


# Computations state
dataSTATE = Dict{String, Any}(
            "state" => nothing,
            "brain" => nothing);


# Dictionary with commands
dicCODE = Dict{String, Functor}()


# Node terminating computation

dicCODE["return the character!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = d["char"]; d),
        Dict(:in => ["chat"], :out => ["res"]))

dicCODE["return its transliteration!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = d["txt"]; d),
        Dict(:in => ["txt"], :out => ["res"]))


dicCODE["is the character found in the mapping file?"] =
    Functor((d,e=nothing,f=nothing) ->
        (!haskey(d, "ix") ?
            begin
                d["v_chars"] = collect(d["txt"])
                d["n_chars"] = length(d["v_chars"])
                d["ix"] = 1;
            end : "";
         d["state"] = haskey(dISOMapping, d["v_chars"][d["ix"]]) ? "yes" : "no";
         d),
        Dict(:in => ["txt"], :out => ["v_chars", "n_chars", "ix"]))

# is it ?

dicCODE["is it ا?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["v_chars"][d["ix"]] == "ا" ? "yes" : "no"; d),
            Dict(:in => ["v_chars", "ix"], :out => ["state"]))

dicCODE["is it ە?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["v_chars"][d["ix"]] == "ە" ? "yes" : "no"; d),
            Dict(:in => ["v_chars", "ix"], :out => ["state"]))

# next characters

dicCODE["is the next character ٔ?"] =
Functor((d,e=nothing,f=nothing) ->
    (d["state"] =
        if d["n_chars"] > d["ix"]
            d["v_chars"][d["ix"]+1] == " ٔ" ? "yes" : "no";
        else
            "no"
        end; d),
        Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["state"]))

dicCODE["is the next character ل?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["v_chars"]d["ix"] == "ل" ? "yes" : "no"; d),
            Dict(:in => ["v_chars", "ix"], :out => ["state"]))




# string('b', '͟')




#===#
dicCODE["change all instances of ي and ك and ۀ in the text to ی and ک and هٔ"] =
    #===
    d_corrects = {'ي' : 'ی',
                  'ك' : 'ک',
                 'ۀ' : 'هٔ'}
    ===#
    Functor((d,e=nothing,f=nothing) ->
        begin
            d["txt"] = replace(d["txt"], 'ك'=>
                                         'ک',
                                            'ي'=>
                                            'ی',
                                               "ۀ" =>
                                                "هٔ")
            d
        end,
        Dict(:in => ["txt"], :out => ["txt"]))
#===#

dicCODE["is it ّ?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["char"] == ''' ?
            "yes" : "no"; d),
        Dict(:in => ["char"], :out => ["state"]))
