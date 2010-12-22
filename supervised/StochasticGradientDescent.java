import java.util.Random;

public class StochasticGradientDescent
{
    public static final double X0 = 1;

    public static void main(String[] args)
    {
	StochasticGradientDescent algo = new StochasticGradientDescent();	
	algo.run_it();
    }
    
    
    public void run_it()
    {
	double[] theta = {1.5, -0.4, 0.85, -4.2};

	int SAMPLES = 10000;
	double[][] x = new double[SAMPLES][4];
	for (int i = 0; i < x.length; i ++) {
	    x[i][0] = this.X0;
	    x[i][1] = Math.random() * 10;
	    x[i][2] = Math.random() * 15;
	    x[i][3] = Math.random() * 40;
	}
	
	/*
	double[][] x = {
	    {this.X0, 0, 2, 4},
	    {this.X0, 1, 3, 5},
	    {this.X0, 2, 2, 7},
	    {this.X0, 3, 5, 0},
	    {this.X0, 4, 1, 8},
	    {this.X0, 5, 4, 4},
	    {this.X0, 6, 5, 1},
	    {this.X0, 7, 2, 3},
	    {this.X0, 8, 3, 2},
	    {this.X0, 9, 7, 6}
	};
	*/

	//System.out.println(get_matrix_string(x));
	double[] y = generate_trainingset_y(theta, x);
	//System.out.println(get_array_string(y));
	// now introduce some randomness on y
	Random rand = new Random();
	for (int i = 0; i < y.length; i ++) {
	    y[i] = (1 + rand.nextDouble() * 0.02) * y[i]; // vary up to 2%
	    //y[i] += rand.nextDouble() * 3;
	}
	//System.out.println(get_array_string(y));

	int num_features = x[0].length;
	int num_samples  = x.length;

	// now the test
	double alpha = 0.002;
	theta[0] = 1000; // init to some random value
	theta[1] = 1000;
	theta[2] = 10;
	theta[3] = 10;
	for (int iter = 0; iter < 50000; iter ++) {
	    double[] new_theta = new double[num_features];
	    for (int i = 0; i < theta.length; i ++) {
		new_theta[i] = theta[i];
	    }

	    for (int i = 0; i < num_samples; i ++) {
		double err = y[i] - compute_h(theta, x[i]);
		//System.out.println("err " + err);
		for (int j = 0; j < num_features; j ++) {
		    theta[j] = theta[j] + alpha * err * x[i][j];
		}
	    }

	    boolean converged = true;
	    for (int j = 0; j < num_features; j ++) {
		if (Math.abs(1 - new_theta[j] / theta[j]) > 0.00001)
		    converged = false;
	    }
	    if (converged) break;

	    System.out.println("Iteration " + iter + ": " + get_array_string(theta));
	}
	    
    }
    
    public double[] generate_trainingset_y(double[] theta, double[][] x)
    {
	// on row of x is x0, x1, x2, ..., xn
	// each row is a new sample, so
	//      number of samples = number of rows

	double[] y = new double[x.length];
	for (int i = 0; i < x.length; i ++) {
	    y[i] = compute_h(theta, x[i]);
	}
	return y;
    }
    
    public double compute_h(double[] theta, double[] x)
    {
	double h = 0;
	for (int i = 0; i < theta.length; i ++) {
	    h += theta[i] * x[i];
	}
	return h;
    }
    
    public String get_matrix_string(double[][] a)
    {
	StringBuffer buffer = new StringBuffer();
	for (int i = 0; i < a.length; i ++) {
	    for (int j = 0; j < a[i].length; j ++) {
		buffer.append(a[i][j]);
		buffer.append('\t');
	    }
	    buffer.append('\n');
	}
	return buffer.toString();
    }

    public String get_array_string(double[] a)
    {
	StringBuffer sb = new StringBuffer();
	for (int i = 0; i < a.length; i ++) {
	    sb.append(a[i]);
	    sb.append('\t');
	}
	return sb.toString();
    }
}
