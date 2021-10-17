package opt.wchen607;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesman {
    /** The n value */
    private static final int N = 30;

    public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }
    public static void main(String[] args) {


        double start, end, time;
        int[] iters = {10,100,500,1000,2500};
        //int[] iters = {10};
        int testRuns = 10;

        for (int iter : iters) {

            double sum_rhc = 0;
            double sum_sa = 0;
            double sum_ga = 0;
            double sum_mimic = 0;

            double time_rhc = 0;
            double time_sa = 0;
            double time_ga = 0;
            double time_mimic = 0;

            for (int j = 0; j < testRuns; j++) {
                Random random = new Random();
                // create the random points
                double[][] points = new double[N][2];
                for (int i = 0; i < points.length; i++) {
                    points[i][0] = random.nextDouble();
                    points[i][1] = random.nextDouble();
                }
                // for rhc, sa, and ga we use a permutation based encoding
                TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
                Distribution odd = new DiscretePermutationDistribution(N);
                NeighborFunction nf = new SwapNeighbor();
                MutationFunction mf = new SwapMutation();
                CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);


                start = System.nanoTime();
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_rhc += ef.value(rhc.getOptimal());
                time_rhc += time;
                //System.out.println("rhc: " + ef.value(rhc.getOptimal()));
                //System.out.println(time);

                start = System.nanoTime();
                SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .85, hcp);
                fit = new FixedIterationTrainer(sa, iter);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_sa += ef.value(sa.getOptimal());
                time_sa += time;
                //System.out.println("sa: " + ef.value(sa.getOptimal()));
                //System.out.println(time);

                start = System.nanoTime();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(100, 50, 30, gap);
                fit = new FixedIterationTrainer(ga, iter);
                fit.train();
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_ga += ef.value(ga.getOptimal());
                time_ga += time;
                //System.out.println("ga: " + ef.value(ga.getOptimal()));
                //System.out.println(time);

                start = System.nanoTime();
                // for mimic we use a sort encoding
                ef = new TravelingSalesmanSortEvaluationFunction(points);
                int[] ranges = new int[N];
                Arrays.fill(ranges, N);
                odd = new  DiscreteUniformDistribution(ranges);
                Distribution df = new DiscreteDependencyTree(.9, ranges);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

                MIMIC mimic = new MIMIC(100, 50, pop);
                fit = new FixedIterationTrainer(mimic, iter);
                fit.train();
                //System.out.println(ef.value(mimic.getOptimal()));
                end = System.nanoTime();
                time = end - start;
                time /= Math.pow(10, 9);
                sum_mimic += ef.value(mimic.getOptimal());
                time_mimic += time;
                //System.out.println("Mimic: " + ef.value(mimic.getOptimal()));
                //System.out.println(time);
            }

            double average_rhc = sum_rhc / testRuns;
            double average_sa = sum_sa / testRuns;
            double average_ga = sum_ga / testRuns;
            double average_mimic = sum_mimic / testRuns;

            double averagetime_rhc = time_rhc / testRuns;
            double averagetime_sa = time_sa / testRuns;
            double averagetime_ga = time_ga / testRuns;
            double averagetime_mimic = time_mimic / testRuns;

            System.out.println("##############");
            System.out.println("this is iteration " + iter);
            System.out.println("rhc average is " + average_rhc + ", time average is " + averagetime_rhc);
            System.out.println("sa average is " + average_sa + ", time average is " + averagetime_sa);
            System.out.println("ga average is " + average_ga + ", time average is " + averagetime_ga);
            System.out.println("mimic average is " + average_mimic + ", time average is " + averagetime_mimic);

            String final_result = "";
            final_result = "rhc" + "," + iter + "," + Double.toString(average_rhc) + "," + Double.toString(averagetime_rhc) + "," +
                    "sa" + "," + iter + "," + Double.toString(average_sa) + "," + Double.toString(averagetime_sa) + "," +
                    "ga" + "," + iter + "," + Double.toString(average_ga) + "," + Double.toString(averagetime_ga) + "," +
                    "mimic" + "," + iter + "," + Double.toString(average_mimic) + "," + Double.toString(averagetime_mimic);

            write_output_to_file("Optimization_Results", "travelingsalesman_results.csv", final_result, true);
        }
    }

}
