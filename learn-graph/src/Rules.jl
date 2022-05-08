
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
        (d["res"] = d["txt"]; d),
        Dict(:in => ["txt"], :out => ["res"]))

dicCODE["return its transliteration!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = d["txt"]; d),
        Dict(:in => ["txt"], :out => ["res"]))

dicCODE["is the character found in the mapping?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = haskey(dISOMapping, d["char"]) ? "yes" : "no"; d),
        Dict(:in => ["char"], :out => ["state"]))

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

dicCODE["is it ّ?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["char"] == ''' ?
            "yes" : "no"; d),
        Dict(:in => ["char"], :out => ["state"]))
