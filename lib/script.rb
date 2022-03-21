
# RUNS:
# ruby script.rb --model_path=../data/transformer.onnx
#

require_relative "transliterator/transformer"
require_relative "transliterator/transliterator"
require_relative "transliterator/encoders"


require 'optparse'
require 'onnxruntime'
require 'yaml'
require 'tqdm'


def parser
  options = {}
  required_args = [:text, :model_path]
  OptionParser.new do |opts|
    opts.banner = "Usage: ruby_onnx.rb [options]"

    #opts.on("-tTEXT", "--text=TEXT", "text to diacritize") do |t|
    #  options[:text] = t
    #end
    #opts.on("-fFILE", "--text_filename=FILE", "path to file to diacritize") do |f|
    #  options[:text_filename] = f
    #end
    opts.on("-mMODEL", "--model_path=MODEL", "path to onnx model") do |m|
      options[:model_path] = m
    end
    #opts.on("-cCONFIG", "--config=CONFIG", "path to config file") do |c|
    #  options[:config] = c
    #end

  end.parse!

  # required args
  [:model_path].each {|arg| raise OptionParser::MissingArgument, arg if options[arg].nil? }
  # p(options)
  options

end


parser = parser()

config_path = parser.has_key?(:config) ? parser[:config] : "../config/model.yml"
config = YAML.load(File.read(config_path))


transliterator = Transliterator::Transliterator.new(parser[:model_path], config)
#transformer = Transliterator::Transformer.new(parser[:model_path], config)
#encoder = Transliterator::FarsiEncoder.new(config)

txt = "یییییییی"

transliterator.transliterate_mocked_text(txt)
