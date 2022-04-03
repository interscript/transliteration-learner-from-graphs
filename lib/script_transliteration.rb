
# RUNS:
# ruby script_transliteration.rb --text "باز آ باز آ هر آنچه هستی باز آ"
# ruby script_transliteration.rb --text_filename test.txt
#

require_relative "transliterator/transliterator"


require 'optparse'
require 'onnxruntime'
require 'yaml'
require 'tqdm'


def parser
  options = {}
  required_args = [:text, :model_path]
  OptionParser.new do |opts|
    opts.banner = "Usage: ruby_onnx.rb [options]"

    opts.on("-tTEXT", "--text=TEXT", "text to diacritize") do |t|
      options[:text] = t
    end
    opts.on("-fFILE", "--text_filename=FILE", "path to file to diacritize") do |f|
      options[:text_filename] = f
    end

  end.parse!

  # required args
  [].each {|arg| raise OptionParser::MissingArgument, arg if options[arg].nil? }
  # p(options)
  options

end


parser = parser()

config_path = parser.has_key?(:config) ? parser[:config] : "../config/model.yml"
config = YAML.load(File.read(config_path))


transliterator = Transliterator::Transliterator.new(config)


if parser.has_key? :text

  txt = parser[:text]
  p(transliterator.transliterate_text(txt))

elsif parser.has_key? :text_filename

  file_name = parser[:text_filename]
  transliterator.transliterate_file(file_name).map {|t| p(t)}

else

  p("no argument recognised: expecting --text or --text_filename")

end
