

use regex::Regex;
use rayon::prelude::*;
use std::sync::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap as HashMap;

fn get_vocab(corpus: Vec<String>) -> HashMap<String,usize>{
    let mut vocab = HashMap::default();

    for word in corpus{
        *vocab.entry(word).or_insert(0)+=1;
    }
    vocab
}


fn get_stats(vocab: &mut HashMap<String, usize>) -> HashMap<(String,String), usize>{
    let mut pairs = HashMap::default();

    for (word, &freq) in vocab.iter(){
        let symbols: Vec<_> = word.chars().collect();

        for i in 0..symbols.len().saturating_sub(1)  {
            let pair = (symbols[i].to_string(), symbols[i+1].to_string());
            *pairs.entry(pair).or_insert(0)+= freq;
        }
    }
pairs
}

fn get_stats_ids(ids: &Vec<u8>) -> HashMap<(u8, u8), usize>{
    let  counts = Mutex::new(HashMap::default());
    ids.par_windows(2).for_each(|pair| {
        let mut counts = counts.lock().unwrap();
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
    });

    let counts = counts.into_inner().unwrap();
    counts
    
}

fn merge(ids: Vec<u8>, pair: (u8, u8), idx :usize ) -> Vec<u8>{
    let mut new_ids = Vec::with_capacity(ids.len());

   let mut i = 0;
 
    while i < ids.len(){
        if i < ids.len() -1  && ids[i] == pair.0 && ids[i+1] == pair.1{
            new_ids.push(idx as u8);
            i += 2;
        }
        else {
            new_ids.push(ids[i]);  
            i += 1;
    }     

   
    }
    new_ids
}
fn merge_vocab(pair: (String, String), vocab: &mut HashMap<String, usize>) {
    let bigram = pair.0.clone() + &pair.1;
    let new_vocab: HashMap<String, usize> = vocab
        .iter()
        .map(|(word, &freq)| (word.replace(&pair.0, &bigram).replace(&pair.1, ""), freq))
        .collect();
    vocab.clear();
    vocab.extend(new_vocab);
}
#[derive(Clone)]
#[pyclass(module= "BPE", get_all)]
struct BPE{
    vocab : HashMap<String, usize>,
    vocab_num : HashMap<u8,Vec<u8>>,
    merges : HashMap<(u8, u8), usize>
}

#[pymethods]
impl  BPE {
    #[new]
    fn new() -> Self{
        BPE{
            vocab : HashMap::default(),
            vocab_num: HashMap::default(),
            merges: HashMap::default(),
        }
    }
    fn parallel_word_tokenizer(&self, raw_corpus: &str)->  Vec<u8>{

        let re: Regex = Regex::new(r"(\b\w+\b|\s+)").unwrap();
        

        let tokens: Vec<u8> = raw_corpus.par_split_inclusive('\n')
        .map(|chunk| {
            re.find_iter(chunk)
                .flat_map(|m| m.as_str().bytes())
        })
        .flatten_iter().collect();
    tokens
    }


    fn parallel_word_tokenizer_2<'a>(&self,raw_corpus: &'a str) -> Vec<&'a str> {
        let re = Regex::new(r"(\b\w+\b|\s+)").unwrap();
        let chunk_size = 10000; // Adjust the chunk size as needed
    
        let chunks: Vec<&str> = raw_corpus
            .split(|c: char| !c.is_alphanumeric())
            .collect();
    
        let tokens: Vec<Vec<&str>> = chunks
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk
                    .iter()
                    .flat_map(|&chunk| re.find_iter(chunk))
                    .map(|m| m.as_str())
                    .collect()
            })
            .collect();
    
        tokens.into_iter().flatten().collect()
    }
    fn word_tokenizer(&self, raw_corpus: String) -> Vec<String>{
        let re = Regex::new(r"(\b\w+\b|\s+)").unwrap();
    
        let tokens : Vec<String> = re.find_iter(raw_corpus.as_str()).map(|m| m.as_str().to_string()).collect();
        tokens
    
    }
    fn str2token(&self, text: Vec<String>) -> Vec<u8>{
        //let tokens: Vec<u8> = text.into_par_iter().map(|m| m.into_bytes()).flatten().collect(); parallel
        let tokens: Vec<u8> = text.iter().flat_map(|m| m.bytes()).collect();

    tokens
    }
  

    fn convert_to_pydict(&self,vocab: HashMap<String, usize>, py: Python) -> PyResult<Py<PyDict>> {
        let py_dict = PyDict::new(py);
        
        for (key, value) in vocab {
            py_dict.set_item(key, value)?;
        }
        
        Ok(py_dict.into_py(py))
    }
    

    fn learn(&mut self,corpus: Vec<String>, epochs: usize,py : Python)-> PyResult<Py<PyDict>>{
        let mut vocab = get_vocab(corpus);
        for _ in 0..epochs{
        let pairs = get_stats(&mut vocab);
        self.vocab = vocab.clone();
        if let Some((max,_))= pairs.into_iter().max_by_key(|&(_, count)| count){

            merge_vocab(max, &mut vocab);
          } 
        }
        return  self.convert_to_pydict(vocab, py);
    }


    fn learn_ids(&mut self,raw_corpus: &str, epochs: usize){
        
        let mut ids = self.parallel_word_tokenizer(raw_corpus);

        let mut merges = HashMap::default();
        let mut vocab = HashMap::default();

        for idx in 0..256{
            let byte=  idx as u8;
            vocab.insert(idx as u8, vec![byte]);
            
        }

         for i in 0..epochs{
         let pairs = get_stats_ids(&ids);

         if let Some(pair)= pairs.keys().max_by_key(|p| pairs.get(p)){
             let idx  = 256 + i;
             ids = merge(ids, *pair,idx);
             merges.insert(*pair, idx);
             let p0_bytes = vocab[&pair.0].clone();
             let p1_bytes = vocab[&pair.1].clone();
             let merged_bytes = [p0_bytes, p1_bytes].concat();
             vocab.insert(idx as u8,merged_bytes);
           }
         }
        self.merges = merges;
        self.vocab_num = vocab;
        

    
}

    fn encode(&mut self ,raw_corpus: String) -> Vec<u8> {
        let text_tokens = self.word_tokenizer(raw_corpus);
        let mut tokens = self.str2token(text_tokens);
        while tokens.len() >= 2{
            let stats = get_stats_ids(&tokens);
            if let Some(pair) = stats.keys().min_by_key(|p| self.merges.get(p)) {
                if !self.merges.contains_key(pair) {
                    break;
                }
                let idx = self.merges.get(pair).unwrap();
                tokens = merge(tokens, *pair, *idx );
            }
        }
    tokens
    }
     fn decode(&self, ids: Vec<u8>) -> String {
        
         let tokens: Vec<u8> = ids.iter().map(|&idx| self.vocab_num[&idx].as_slice()).flatten().cloned().collect();

         let text = String::from_utf8_lossy(&tokens).to_string();
         text
    }
    fn decode_bpe_tokens(&mut self,tokens: Vec<u8>) -> String {
        let mut text = tokens.iter().map(|&token| token as char).collect::<String>();
    
        // Reverse BPE merges to reconstruct original text
        for ((merge_from_a, merge_from_b), &merge_to) in self.merges.iter() {
            let merge_from = format!("{}{}", *merge_from_a as char, *merge_from_b as char);
            let merge_to_char = merge_to as u8 as char;
            text = text.replace(merge_to_char, &merge_from);
        }
    
        text
    }
}



/// A Python module implemented in Rust.
#[pymodule]
fn tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BPE>()?;

    Ok(())
}
