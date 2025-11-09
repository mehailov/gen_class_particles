#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <numeric>
#include "TChain.h"
#include "TFile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TRandom.h"
#include "TGraph.h"
#include <TLegend.h>

using namespace std;
//==============================================================================================
// Класс нейрона
//==============================================================================================
class neuron {
public:
    float shift;
    float shift_opt_step;
    int is_alive;
    float input;
    float output;
    float mut_prob;
    int layer_type; 
    void get_output();
    
    void copy(neuron n0) {
        this->shift = n0.shift;
        this->shift_opt_step = n0.shift_opt_step;
        this->is_alive = n0.is_alive;
        this->input = n0.input;
        this->output = n0.output;
        this->mut_prob = n0.mut_prob;
        this->layer_type =n0.layer_type;
    }
};

void neuron::get_output() {
    // Для скрытых слоев
    if (layer_type == 0) {
        this->output = fmax(0.0f, this->input) + this->shift;
    }
    // Для выходного слоя
    else {
        this->output = fmax(0.0f, this->input) + this->shift;
    }
}
//==============================================================================================
// Класс синапса
//==============================================================================================
class synapse {
public:
    float weight;
    float weight_opt_step;
    float mut_prob;

    void copy(synapse s0) {
        this->weight = s0.weight;
        this->weight_opt_step = s0.weight_opt_step;
        this->mut_prob = s0.mut_prob;
    }
};

//==============================================================================================
// Класс слоя нейронов
//==============================================================================================
class Layer {
public:
    vector<neuron> neurons;

    void initiate_Layer(int n_neurons,int layer_type = 0) {
        neuron n0;
        for (int k = 0; k < n_neurons; k++) {
            n0.shift_opt_step = 0.03;
            n0.is_alive = 1;
            n0.output = 0.0;
            n0.input = 0.0;
            n0.layer_type = layer_type;
            if (layer_type == 1) {  //
             n0.shift = 0.0f;      // Без сдвига
             n0.mut_prob = 0.0f;   // Без мутаций
         } else {
             n0.shift = 0.02 * gRandom->Gaus(0, 1);
             n0.mut_prob = 0.02f;
         }
            this->neurons.push_back(n0);
        }
    }

    void copy(Layer l0) {
        this->neurons.clear();
        for (size_t k = 0; k < l0.neurons.size(); k++) {
            neuron n0;
            n0.copy(l0.neurons[k]);
            this->neurons.push_back(n0);
        }
    }

    void mutation_Layer() {
        random_device rd;
        mt19937 genfnum(rd());
        uniform_real_distribution<float> rnd01(0.0, 1.0);

        for (size_t k = 0; k < neurons.size(); k++) {
            float r = rnd01(genfnum);
            if (r > neurons[k].mut_prob) continue;

            neurons[k].shift += neurons[k].shift_opt_step * gRandom->Gaus(0, 1);
            float fact = 1.0 + 0.01 * gRandom->Gaus(0, 1);
            if (r < neurons[k].mut_prob / 10.0) fact += 0.3 * gRandom->Gaus(0, 1);
            if (fact > 1.5) fact = 1.5;
            if (fact < 0.66) fact = 0.66;
            neurons[k].shift_opt_step *= fact;
            neurons[k].is_alive = 1;
            neurons[k].output = 0.0;
            neurons[k].input = 0.0;
        }
    }
};//==============================================================================================
// Класс слоя синапсов
//==============================================================================================
class SynapseLayer {
public:
    vector<vector<synapse>> synapses;

    void initiate_Synapses(int n_neurons0, int n_neurons1) {
        for (int k = 0; k < n_neurons0; k++) {
            vector<synapse> synp0;
            for (int p = 0; p < n_neurons1; p++) {
                synapse s0;
                s0.weight = 0.1 + 0.02 * gRandom->Gaus(0, 1);
                if (p % 2 == 0) s0.weight = -0.2;
                if (p % 3 == 0) s0.weight = 0.3;
                s0.weight_opt_step = 0.05;
                s0.mut_prob = 0.01;
                synp0.push_back(s0);
            }
            this->synapses.push_back(synp0);
        }
    }

    void copy(SynapseLayer SL0) {
        this->synapses.clear();
        for (size_t k = 0; k < SL0.synapses.size(); k++) {
            vector<synapse> synp0;
            for (size_t p = 0; p < SL0.synapses[k].size(); p++) {
                synapse s0;
                s0.copy(SL0.synapses[k][p]);
                synp0.push_back(s0);
            }
            this->synapses.push_back(synp0);
        }
    }

    void mutation_SynapseLayer() {
        random_device rd;
        mt19937 genfnum(rd());
        uniform_real_distribution<float> rnd01(0.0, 1.0);

        for (size_t k = 0; k < synapses.size(); k++) {
            for (size_t p = 0; p < synapses[k].size(); p++) {
                float r = rnd01(genfnum);
                if (r > synapses[k][p].mut_prob) continue;

                synapses[k][p].weight += synapses[k][p].weight_opt_step * gRandom->Gaus(0, 1);
                float fact = 1.0 + 0.01 * gRandom->Gaus(0, 1);
                if (r < synapses[k][p].mut_prob / 10.0) fact += 0.3 * gRandom->Gaus(0, 1);
                if (fact > 1.5) fact = 1.5;
                if (fact < 0.66) fact = 0.66;
                synapses[k][p].weight_opt_step *= fact;
            }
        }
    }
};

//==============================================================================================
// Класс нейронной сети
//==============================================================================================
class Deep_Network {
public:
    vector<Layer> layers;
    vector<SynapseLayer> synapse_connections;
    float fitness;

    void propagare_weight() {
        for (size_t k = 0; k < layers.size() - 1; k++) {
            for (size_t n = 0; n < layers[k + 1].neurons.size(); n++)
                layers[k + 1].neurons[n].input = 0.0;

            for (size_t n = 0; n < layers[k].neurons.size(); n++)
                layers[k].neurons[n].get_output();

            for (size_t s = 0; s < synapse_connections[k].synapses.size(); s++) {
                for (size_t s1 = 0; s1 < synapse_connections[k].synapses[s].size(); s1++) {
                    float weightn = layers[k].neurons[s].output;
                    layers[k + 1].neurons[s1].input += weightn * synapse_connections[k].synapses[s][s1].weight;
                }
            }
        }
        for (size_t n = 0; n < layers.back().neurons.size(); n++)
            layers.back().neurons[n].get_output();
    }

    void copy(Deep_Network nn0) {
        this->layers.clear();
        this->synapse_connections.clear();
        this->fitness = nn0.fitness;

        for (size_t l = 0; l < nn0.layers.size(); l++) {
            Layer l00;
            l00.copy(nn0.layers[l]);
            this->layers.push_back(l00);
        }

        for (size_t sl = 0; sl < nn0.synapse_connections.size(); sl++) {
            SynapseLayer sl00;
            sl00.copy(nn0.synapse_connections[sl]);
            this->synapse_connections.push_back(sl00);
        }
    }

    void mutation() {
        for (auto& layer : layers) {
            layer.mutation_Layer();
        }
        for (auto& syn_layer : synapse_connections) {
            syn_layer.mutation_SynapseLayer();
        }
    }
};
//==============================================================================================
// Вспомогательные функции
//==============================================================================================
void setup_network(Deep_Network& network) {
    Layer l0, l1, l11, l2;
    SynapseLayer sl0, sl01, sl1;

    l0.initiate_Layer(12,0);
    l1.initiate_Layer(24,0);
    l11.initiate_Layer(12,0);
    l2.initiate_Layer(1,1);
    for (auto& neuron : l0.neurons) neuron.mut_prob = 0.2f;
    for (auto& neuron : l1.neurons) neuron.mut_prob = 0.2f;
    for (auto& neuron : l11.neurons) neuron.mut_prob = 0.2f;
    sl0.initiate_Synapses(12, 24);
    sl01.initiate_Synapses(24, 12);
    sl1.initiate_Synapses(12, 1);
    for (auto& row : sl0.synapses) {
        for (auto& syn : row) syn.mut_prob = 0.1f;
    }
    for (auto& row : sl1.synapses) {
        for (auto& syn : row) syn.mut_prob = 0.1f;
    }
    network.layers.push_back(l0);
    network.layers.push_back(l1);
    network.layers.push_back(l11);
    network.layers.push_back(l2);

    network.synapse_connections.push_back(sl0);
    network.synapse_connections.push_back(sl01);
    network.synapse_connections.push_back(sl1);
}

vector<vector<float>> read_data(const char* filename, const char* treename) {
    TChain chain(treename);
    chain.Add(filename);
    
    Int_t nHits, nHitsTsB, nHitsTsEC, nPoints, nHitsRS;
    Float_t pt, ptot, eta, phi, trk_length, trk_length_RS, chi2NDF;
    
    chain.SetBranchAddress("nHits", &nHits);
    chain.SetBranchAddress("nHitsTsB", &nHitsTsB);
    chain.SetBranchAddress("nHitsTsEC", &nHitsTsEC);
    chain.SetBranchAddress("nPoints", &nPoints);
    chain.SetBranchAddress("pt", &pt);
    chain.SetBranchAddress("ptot", &ptot);
    chain.SetBranchAddress("eta", &eta);
    chain.SetBranchAddress("phi", &phi);
    chain.SetBranchAddress("trk_length", &trk_length);
    chain.SetBranchAddress("trk_length_RS", &trk_length_RS);
    chain.SetBranchAddress("nHitsRS", &nHitsRS);
    chain.SetBranchAddress("chi2NDF", &chi2NDF);

    vector<vector<float>> data;
    Long64_t nEntries = chain.GetEntries();
    
    for (Long64_t i = 0; i < nEntries; i++) {
        chain.GetEntry(i);
        vector<float> track_data = {
            static_cast<float>(nHits),
            static_cast<float>(nHitsTsB),
            static_cast<float>(nHitsTsEC),
            static_cast<float>(nPoints),
            pt, ptot, eta, phi,
            trk_length, trk_length_RS,
            static_cast<float>(nHitsRS),
            chi2NDF
        };
        data.push_back(track_data);
    }
    return data;
}

pair<vector<vector<float>>, vector<vector<float>>> split_data(
    const vector<vector<float>>& data, float train_ratio = 0.8) {
    
    vector<vector<float>> train_data, test_data;
    vector<vector<float>> shuffled_data = data;
    
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(shuffled_data.begin(), shuffled_data.end(), default_random_engine(seed));
    
    size_t split_index = static_cast<size_t>(shuffled_data.size() * train_ratio);
    
    for (size_t i = 0; i < shuffled_data.size(); i++) {
        if (i < split_index) train_data.push_back(shuffled_data[i]);
        else test_data.push_back(shuffled_data[i]);
    }
    
    return make_pair(train_data, test_data);
}
float calculate_fitness(Deep_Network& net,
                       const vector<vector<float>>& pions_train,
                       const vector<vector<float>>& muons_train) {
    float fitness = 0.0f;

    // Предварительно вычисляем средние значения выходов
    float pion_output_sum = 0.0f;
    float muon_output_sum = 0.0f;
    int pion_count = 0;
    int muon_count = 0;

    // Первый проход: вычисляем средние значения
    for (const auto& track : pions_train) {
        for (size_t i = 0; i < track.size(); ++i) {
            net.layers[0].neurons[i].input = track[i];
        }
        net.propagare_weight();
        float output = net.layers.back().neurons[0].output;
        pion_output_sum += output;
        pion_count++;
    }
    
    for (const auto& track : muons_train) {
        for (size_t i = 0; i < track.size(); ++i) {
            net.layers[0].neurons[i].input = track[i];
        }
        net.propagare_weight();
        float output = net.layers.back().neurons[0].output;
        muon_output_sum += output;
        muon_count++;
    }
    
    float mean_pion = (pion_count > 0) ? pion_output_sum / pion_count : 0.2f;
    float mean_muon = (muon_count > 0) ? muon_output_sum / muon_count : 0.8f;
    float threshold = (mean_muon+mean_pion)/2;

    // Второй проход: вычисляем фитнес с учетом средних значений
    for (const auto& track : pions_train) {
        for (size_t i = 0; i < track.size(); ++i) {
            net.layers[0].neurons[i].input = track[i];
        }
        net.propagare_weight();
        float output = net.layers.back().neurons[0].output;
        
        if (output < threshold) {
            // Награда за правильную классификацию
            float diff = output - mean_pion;
            float DIFF= -output+threshold;
            fitness += 2.0f*DIFF - pow(abs(diff),2);  //
        } else {
            // Штраф за ложноположительный
            float diff = output - mean_pion;
            fitness -= 2.0f*pow(abs(diff),2);  // Штрафуем за отклонение от среднего
        }
    }

    for (const auto& track : muons_train) {
        for (size_t i = 0; i < track.size(); ++i) {
            net.layers[0].neurons[i].input = track[i];
        }
        net.propagare_weight();
        float output = net.layers.back().neurons[0].output;
        
        if (output >= threshold) {
            // Награда за правильную классификацию
            float diff = output - mean_muon;
            float DIFF= output-threshold;
            fitness += DIFF- pow(abs(diff),2);  // 
        } else {
            // Штраф за ложноотрицательный
            float diff = output - mean_muon;
            fitness -= pow(abs(diff),2);  // Штрафуем за отклонение от среднего
        }
    }

    // Нормализация по размеру выборки
    fitness /= (pions_train.size() + muons_train.size());
    
    return fitness;
}

void evaluate_network(Deep_Network& net,
                      const vector<vector<float>>& pions_data,
                      const vector<vector<float>>& muons_data) {
    int correct_pions = 0;
    int correct_muons = 0;
    float pion_output_sum = 0.0f;
    float muon_output_sum = 0.0f;
    int pion_count = 0;
    int muon_count = 0;
    //threshold new
    for (const auto& track : pions_data) {
        for (size_t i = 0; i < track.size(); ++i) {
            net.layers[0].neurons[i].input = track[i];
        }
        net.propagare_weight();
        float output = net.layers.back().neurons[0].output;
        pion_output_sum += output;
        pion_count++;
    }
    
    for (const auto& track : muons_data) {
        for (size_t i = 0; i < track.size(); ++i) {
            net.layers[0].neurons[i].input = track[i];
        }
        net.propagare_weight();
        float output = net.layers.back().neurons[0].output;
        muon_output_sum += output;
        muon_count++;
    }
    float mean_pion = (pion_count > 0) ? pion_output_sum / pion_count : 0.2f;
    float mean_muon = (muon_count > 0) ? muon_output_sum / muon_count : 0.8f;
    float threshold = (mean_muon+mean_pion)/2;
    // Пионы
    for (const auto& track : pions_data) {
        for (size_t i = 0; i < track.size(); ++i) {
            net.layers[0].neurons[i].input = track[i];
        }
        net.propagare_weight();
        float output = net.layers.back().neurons[0].output;
        if (output < threshold) correct_pions++;
    }

    // Мюоны
    for (const auto& track : muons_data) {
        for (size_t i = 0; i < track.size(); ++i) {
            net.layers[0].neurons[i].input = track[i];
        }
        net.propagare_weight();
        float output = net.layers.back().neurons[0].output;
        if (output > threshold) correct_muons++;
    }

    float pion_acc = static_cast<float>(correct_pions) / pions_data.size();
    float muon_acc = static_cast<float>(correct_muons) / muons_data.size();
    float total_acc = static_cast<float>(correct_pions + correct_muons) / 
                     (pions_data.size() + muons_data.size());

    cout << "\n===== Evaluation ====="
         << "\nPions accuracy: " << pion_acc * 100 << "% (" 
         << correct_pions << "/" << pions_data.size() << ")"
         << "\nMuons accuracy: " << muon_acc * 100 << "% (" 
         << correct_muons << "/" << muons_data.size() << ")"
         << "\nTotal accuracy: " << total_acc * 100 << "%\n";
}

Deep_Network tournament_selection(const vector<Deep_Network>& population, int tournament_size) {
    vector<size_t> indices(population.size());
    iota(indices.begin(), indices.end(), 0);
    
    shuffle(indices.begin(), indices.end(), mt19937(random_device{}()));
    
    size_t best_idx = indices[0];
    float best_fitness = population[best_idx].fitness;
    
    for (int i = 1; i < tournament_size; i++) {
        if (population[indices[i]].fitness > best_fitness) {
            best_fitness = population[indices[i]].fitness;
            best_idx = indices[i];
        }
    }
    
    return population[best_idx];
}

void crossover(const Deep_Network& parent1, const Deep_Network& parent2, Deep_Network& child) {
    child.copy(parent1);
    random_device rd;
    mt19937 gen(rd());
    
    // Кроссовер нейронов
    for (size_t l = 0; l < child.layers.size(); l++) {
        uniform_int_distribution<size_t> dist(0, child.layers[l].neurons.size() - 1);
        size_t crossover_point = dist(gen);
        
        for (size_t n = crossover_point; n < child.layers[l].neurons.size(); n++) {
            child.layers[l].neurons[n].copy(parent2.layers[l].neurons[n]);
        }
    }
    
    // Кроссовер синапсов
    for (size_t sl = 0; sl < child.synapse_connections.size(); sl++) {
        uniform_int_distribution<size_t> dist(0, child.synapse_connections[sl].synapses.size() - 1);
        size_t crossover_point = dist(gen);
        
        for (size_t k = crossover_point; k < child.synapse_connections[sl].synapses.size(); k++) {
            for (size_t p = 0; p < child.synapse_connections[sl].synapses[k].size(); p++) {
                child.synapse_connections[sl].synapses[k][p].copy(parent2.synapse_connections[sl].synapses[k][p]);
            }
        }
    }
}
void save_network_txt(const Deep_Network& net, const string& filename) {
    ofstream out(filename);
    
    // Сохраняем параметры слоев
    out << net.layers.size() << "\n";
    for (const auto& layer : net.layers) {
        out << layer.neurons.size() << " ";
    }
    out << "\n";
    
    // Сохраняем нейроны
    for (const auto& layer : net.layers) {
        for (const auto& neuron : layer.neurons) {
            out << neuron.shift << " " << neuron.shift_opt_step << " " 
                << neuron.mut_prob << "\n";
        }
    }
    
    // Сохраняем синапсы
    for (const auto& syn_layer : net.synapse_connections) {
        out << syn_layer.synapses.size() << " " 
            << syn_layer.synapses[0].size() << "\n";
            
        for (const auto& row : syn_layer.synapses) {
            for (const auto& syn : row) {
                out << syn.weight << " " << syn.weight_opt_step << " "
                    << syn.mut_prob << " ";
            }
            out << "\n";
        }
    }
    out.close();
}
void create_histograms(const std::vector<float>& pions_outputs, 
                      const std::vector<float>& muons_outputs,
                      int generation,
                      const char* base_filename = "output_histograms") 
{
    // Автоматический расчёт границ
    auto min_max_pions = std::minmax_element(pions_outputs.begin(), pions_outputs.end());
    auto min_max_muons = std::minmax_element(muons_outputs.begin(), muons_outputs.end());
    
    float global_min = std::min(*min_max_pions.first, *min_max_muons.first);
    float global_max = std::max(*min_max_pions.second, *min_max_muons.second);
    
    // Добавляем небольшой отступ
    float margin = 0.1 * (global_max - global_min);
    global_min -= margin;
    global_max += margin;
    
    // Создаём гистограммы с общими границами
    TH1F h_pions("h_pions", "Pions vs Muons;Output value;Counts", 100, global_min, global_max);
    TH1F h_muons("h_muons", "Pions vs Muons;Output value;Counts", 100, global_min, global_max);
    
    for (float output : pions_outputs) h_pions.Fill(output);
    for (float output : muons_outputs) h_muons.Fill(output);
    
    // Настраиваем цвета и прозрачность
    h_pions.SetLineColor(kRed);
    h_pions.SetFillColor(kRed);
    h_pions.SetFillStyle(3003); // Полупрозрачная заливка
    h_muons.SetLineColor(kBlue);
    h_muons.SetFillColor(kBlue);
    h_muons.SetFillStyle(3004);

    // Создаём canvas и рисуем гистограммы вместе
    TCanvas canvas("canvas", "Pions vs Muons", 800, 600);
    
    // Рисуем первую гистограмму и настраиваем оси
    h_pions.Draw();
    h_muons.Draw("SAME"); // Ключевая опция: SAME (наложение)
    // Добавляем легенду
    TLegend legend(0.7, 0.7, 0.9, 0.9);
    legend.AddEntry(&h_pions, "Pions", "f");
    legend.AddEntry(&h_muons, "Muons", "f");
    legend.Draw();

    // Сохраняем в файл
    std::string filename = std::string(base_filename) + "_gen" + std::to_string(generation) + ".png";
    canvas.SaveAs(filename.c_str());
}

void train_network(int generations, int population_size,
                   const vector<vector<float>>& pions_data,
                   const vector<vector<float>>& muons_data) 
{
    // Разделение данных
    auto [pions_train, pions_test] = split_data(pions_data, 0.8);
    auto [muons_train, muons_test] = split_data(muons_data, 0.8);
    
    cout << "Data split complete:\n"
         << "Pions train: " << pions_train.size() << ", test: " << pions_test.size() << "\n"
         << "Muons train: " << muons_train.size() << ", test: " << muons_test.size() << endl;
    ofstream acc_file("accuracy.txt");
acc_file << "Generation Threshold TrainPion TrainMuon TestPion TestMuon\n";
    // Инициализация популяции
    vector<Deep_Network> population;
    for (int i = 0; i < population_size; i++) {
        Deep_Network net;
        setup_network(net);
        population.push_back(net);
    }
    
    const int tournament_size = 5;
    const float elite_percent = 0.1f;
    const int elite_count = max(1, static_cast<int>(population_size * elite_percent));
    
    Deep_Network best_network;
    float best_fitness = -numeric_limits<float>::max();
    vector<float> best_fitness_history;
    
    // Основной цикл обучения
    for (int gen = 0; gen < generations; gen++) {
        // Расчет приспособленности
        float total_fitness = 0.0f;
        for (auto& net : population) {
            net.fitness = calculate_fitness(net, pions_train, muons_train);
            total_fitness += net.fitness;
            
            if (net.fitness > best_fitness) {
                best_fitness = net.fitness;
                best_network.copy(net);
            }
        }
        best_fitness_history.push_back(best_fitness);
        if (gen % 10 == 0) {
    float pion_out = 0, muon_out = 0;
    
    for (auto& track : pions_train) {
        for (size_t i = 0; i < track.size(); i++) {
            best_network.layers[0].neurons[i].input = track[i];
        }
        best_network.propagare_weight();
        pion_out += best_network.layers.back().neurons[0].output;
    }
    
    for (auto& track : muons_train) {
        for (size_t i = 0; i < track.size(); i++) {
            best_network.layers[0].neurons[i].input = track[i];
        }
        best_network.propagare_weight();
        muon_out += best_network.layers.back().neurons[0].output;
    }
    
    float threshold = ((pion_out/pions_train.size()) + (muon_out/muons_train.size()))/2.0f;
    auto calc_accuracy = [&](const auto& tracks, bool is_pion) {
        int correct = 0;
        for (auto& track : tracks) {
            for (size_t i = 0; i < track.size(); i++) {
                best_network.layers[0].neurons[i].input = track[i];
            }
            best_network.propagare_weight();
            float output = best_network.layers.back().neurons[0].output;
            if ((is_pion && output < threshold) || (!is_pion && output >= threshold)) {
                correct++;
            }
        }
        return (float)correct/tracks.size();
    };
    acc_file << gen << " " << threshold << " "
             << calc_accuracy(pions_train, true) << " "
             << calc_accuracy(muons_train, false) << " "
             << calc_accuracy(pions_test, true) << " "
             << calc_accuracy(muons_test, false) << "\n";
}
        // Сортировка популяции
        sort(population.begin(), population.end(),
            [](const Deep_Network& a, const Deep_Network& b) {
                return a.fitness > b.fitness;
            });
            // Периодическая оценка и сохранение гистограмм
        if (gen % 10 == 0) {
            cout << "\nGeneration " << gen << " - Evaluating best network:";
            evaluate_network(best_network, pions_test, muons_test);
            
            // Сбор выходов сети для тестовых данных
            vector<float> pions_outputs;
            for (const auto& track : pions_test) {
                for (size_t i = 0; i < track.size(); ++i) {
                    best_network.layers[0].neurons[i].input = track[i];
                }
                best_network.propagare_weight();
                pions_outputs.push_back(best_network.layers.back().neurons[0].output);
            }
            
            vector<float> muons_outputs;
            for (const auto& track : muons_test) {
                for (size_t i = 0; i < track.size(); ++i) {
                    best_network.layers[0].neurons[i].input = track[i];
                }
                best_network.propagare_weight();
                muons_outputs.push_back(best_network.layers.back().neurons[0].output);
            }
            
            // Построение гистограмм
            create_histograms(pions_outputs, muons_outputs, gen);
        }
        
        cout << "Generation " << gen
             << ", Best fitness: " << best_fitness
             << ", Avg fitness: " << total_fitness / population_size << endl;
        
        // Создание нового поколения
        vector<Deep_Network> new_population;
        
        // Элитизм
        for (int i = 0; i < elite_count; i++) {
            new_population.push_back(population[i]);
        }
        
        // Генерация потомков
        while (new_population.size() < static_cast<size_t>(population_size)) {
            Deep_Network parent1 = tournament_selection(population, tournament_size);
            Deep_Network parent2 = tournament_selection(population, tournament_size);
            
            Deep_Network child;
            crossover(parent1, parent2, child);
            child.mutation();
            
            new_population.push_back(child);
        }
        
        population = new_population;
    }
    
    // Финальная оценка
    cout << "\n===== Final Evaluation =====";
    evaluate_network(best_network, pions_test, muons_test);
    
    // Сохранение лучшей сети
    save_network_txt(best_network, "best_network.txt");
    acc_file.close();
    // Сохранение истории фитнеса
    ofstream history_file("fitness_history.txt");
    for (float f : best_fitness_history) {
        history_file << f << "\n";
    }
    history_file.close();
}

void run_trained_network(const char* network_file,
                         const vector<vector<float>>& pions_data,
                         const vector<vector<float>>& muons_data) {
    TFile fin(network_file);
    Deep_Network* best_net = nullptr;
    fin.GetObject("best_network", best_net);
    
    if (!best_net) {
        cerr << "Error: Network not found in file!" << endl;
        return;
    }
    
    vector<float> pions_outputs;
    for (const auto& track : pions_data) {
        for (size_t i = 0; i < track.size(); ++i) {
            best_net->layers[0].neurons[i].input = track[i];
        }
        best_net->propagare_weight();
        pions_outputs.push_back(best_net->layers.back().neurons[0].output);
    }
    
    vector<float> muons_outputs;
    for (const auto& track : muons_data) {
        for (size_t i = 0; i < track.size(); ++i) {
            best_net->layers[0].neurons[i].input = track[i];
        }
        best_net->propagare_weight();
        muons_outputs.push_back(best_net->layers.back().neurons[0].output);
    }
    
    create_histograms(pions_outputs, muons_outputs,300 ,"final_outputs.png");
    fin.Close();
}
void n5() {
    // Загрузка данных
    vector<vector<float>> pions_data = read_data("output_tracks_pion_15_21 (1).root", "tracks");
    vector<vector<float>> muons_data = read_data("output_tracks_muon_15_21 (2).root", "tracks");
    
    // Параметры обучения
    const int generations = 300;
    const int population_size = 100;
    // Обучение сети
    train_network(generations, population_size, pions_data, muons_data);
}
