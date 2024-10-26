use bitvec::prelude::*;
use json::JsonValue;
use plotters::prelude::*;
use polars::{
    lazy::dsl::{col, cols, lit},
    prelude::*,
};
use rand::prelude::*;
use std::collections::HashSet;
use std::fs;

static POPULATION_SIZE: usize = 20; // Keep this a multiple of 4
static GENERATIONS_NUM: usize = 100;

type Chromosome = BitArr!(for 14, in u16, Msb0);

// Plot average fitness values across generations
fn fitness_plotter(summary: &Vec<f64>) {
    let root = BitMapBackend::new("./line_graph.png", (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Chart of Average Fitness values",
            ("sans-serif", 20).into_font(),
        )
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0i32..(GENERATIONS_NUM + 2) as i32, 0f64..50f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Generation Number")
        .y_desc("Average Fitness")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            summary
                .iter()
                .enumerate()
                .map(|(index, &value)| (index as i32, value)),
            &RED,
        ))
        .unwrap();

    root.present().unwrap();
}

// Displays the meaning of the rules
fn rule_interpreter(
    chosen_rules: &Vec<(usize, f64)>,
    population: &Vec<Chromosome>,
    encodings: &JsonValue,
) {
    let indices = [(0, 2), (2, 9), (9, 13), (13, 14)];
    let fields = ["protocol_types", "service_ports", "tcp_flags", "attack"];

    for rule in chosen_rules {
        let member: Chromosome = population[rule.0];
        for (i, (start, end)) in indices.iter().enumerate() {
            let memb_field: &BitSlice<_, _> = &member[*start..*end];
            let memb_dec: u32 = memb_field.load();
            print!("{:15} ", encodings[fields[i]][memb_dec.to_string()]);
        }
        println!();
    }
}

fn fitness(member: &Chromosome, lf: LazyFrame) -> f64 {
    let indices = [(0, 2), (2, 9), (9, 13), (13, 14)];
    let mut values: Vec<u32> = Vec::new();
    for (start, end) in indices {
        let memb_field: &BitSlice<_, _> = &member[start..end];
        let memb_dec: u32 = memb_field.load();
        values.push(memb_dec);
    }
    let outcome = values[3];

    let mask = (col("protocol_type").eq(lit(values[0])))
        .and(col("service").eq(lit(values[1])))
        .and(col("flag").eq(lit(values[2])));
    let df = lf.filter(mask);

    let a: usize = df
        .clone()
        .filter(col("outcome").eq(lit(outcome)))
        .collect()
        .unwrap()
        .shape()
        .0;
    let b: usize = df
        .filter(
            col("outcome").eq(lit(1 - outcome)).or(col("outcome")
                .eq(lit(1 - outcome))
                .or(col("outcome").eq(lit(2)))),
        )
        .collect()
        .unwrap()
        .shape()
        .0;
    let outcome_count = vec![281, 107, 97];
    let a_total: usize = outcome_count[outcome as usize];
    let b_total: usize = outcome_count.iter().sum::<usize>() - a_total;

    let fitness_val: f64 = (a as f64 / a_total as f64) - (b as f64 / b_total as f64);
    fitness_val
}

fn sorter(
    population: &Vec<Chromosome>,
    lf: &LazyFrame,
    summary: &mut Vec<f64>,
) -> Vec<(usize, f64)> {
    let mut fitness_data = Vec::new();
    let mut fitness_val: f64;
    let mut fitness_total: f64 = 0.0;
    for (index, member) in population.iter().enumerate() {
        fitness_val = fitness(member, lf.clone());
        fitness_data.push((index, fitness_val));
        fitness_total += fitness_val;
    }
    fitness_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    summary.push(fitness_total / (POPULATION_SIZE as f64));
    fitness_data
}

fn elitism(
    population: &Vec<Chromosome>,
    lf: &LazyFrame,
    summary: &mut Vec<f64>,
) -> Vec<Chromosome> {
    let mut new_population: Vec<Chromosome> = Vec::new();
    let fitness_data = sorter(population, lf, summary);
    for (index, _fitness_val) in fitness_data[0..(POPULATION_SIZE / 2)].iter() {
        new_population.push(population[*index]);
    }
    new_population
}

fn crossover(population: &Vec<Chromosome>, rng: &mut ThreadRng) -> Vec<Chromosome> {
    let mut crossover_members: Vec<Chromosome> = Vec::new();
    for _count in 0..(POPULATION_SIZE / 4) {
        let mut member1 = population[rng.gen_range(0..population.len())];
        let mut member2 = population[rng.gen_range(0..population.len())];
        let crossover_position = rng.gen_range(1..14);
        let member1_slice = &mut member1.as_mut_bitslice()[crossover_position..14];
        let member2_slice = &mut member2.as_mut_bitslice()[crossover_position..14];

        member1_slice.swap_with_bitslice(member2_slice);
        crossover_members.extend(vec![member1, member2]);
    }
    crossover_members
}

fn mutation(population: &mut Vec<Chromosome>, rng: &mut ThreadRng) {
    //Mutation probability is defined here as 5%
    for member in population.iter_mut() {
        let rand_num: usize = rng.gen_range(1..=100);
        if vec![1, 2, 3, 4, 5].contains(&rand_num) {
            let mutate_position: usize = rng.gen_range(0..14);
            let value = member[mutate_position];
            member.set(mutate_position, !value);
        }
    }
}

// This function checks if member values do not exceed max. If they do, it wraps the values around.
fn member_corrector(population: &mut Vec<Chromosome>) {
    let indices = [(0, 2, 2), (2, 9, 64), (9, 13, 8), (13, 14, 1)];
    for members in population {
        for (start, end, max) in indices {
            let memb_field: &mut BitSlice<_, _> = &mut members[start..end];
            let mut memb_dec: usize = memb_field.load();
            if memb_dec > max {
                memb_dec %= max + 1;
                memb_field.store(memb_dec);
            }
        }
    }
}

fn rand_member_gen(population: &mut Vec<Chromosome>, rng: &mut ThreadRng) {
    let mut chromosome: Chromosome = BitArray::new([0]);
    for _ in population.len()..POPULATION_SIZE {
        loop {
            for i in 0..14 {
                let rand_bit: bool = rng.gen();
                chromosome.set(i, rand_bit);
            }
            if !population.contains(&chromosome) {
                population.push(chromosome);
                break;
            }
        }
    }
}

fn main() {
    let mut population: Vec<Chromosome> = Vec::new();
    let mut summary: Vec<f64> = Vec::new();

    let mut rng = rand::thread_rng();

    // Generating random initial population and correcting values
    rand_member_gen(&mut population, &mut rng);
    member_corrector(&mut population);

    // Import selective columns of the dataset
    let lf = LazyCsvReader::new("./kdd99_10_perc.csv")
        .finish()
        .unwrap()
        .select([cols(vec!["protocol_type", "service", "flag", "outcome"])]);

    // Iterative process of GA starts here
    for _gen_count in 0..GENERATIONS_NUM {
        let mut new_population: Vec<Chromosome> = Vec::new();

        // Elite members of old population are added to new population
        new_population.extend(elitism(&population, &lf, &mut summary));

        // Performing crossover and mutation and adding to new population
        let mut crossover_members = crossover(&population, &mut rng);
        mutation(&mut crossover_members, &mut rng);
        new_population.extend(crossover_members);

        // Keeping only unique members and preserving population size by adding random members
        let unique_population: HashSet<_> = new_population.into_iter().collect();
        new_population = unique_population.into_iter().collect();
        rand_member_gen(&mut new_population, &mut rng);
        member_corrector(&mut new_population);

        // Old population is replaced by new population for next generation
        population = new_population;
    }

    let unique_population: HashSet<_> = population.into_iter().collect();
    let final_population = unique_population.into_iter().collect();
    let final_population_order = sorter(&final_population, &lf, &mut summary);
    let chosen_rules: Vec<(usize, f64)> = final_population_order.into_iter().take(10).collect();

    // Convert binary rule to its original meaning
    let encodings_file = fs::read_to_string("./encodings.json").unwrap();
    let encodings = json::parse(&encodings_file).unwrap();
    rule_interpreter(&chosen_rules, &final_population, &encodings);

    // Plot an average fitness graph
    fitness_plotter(&summary);
}
