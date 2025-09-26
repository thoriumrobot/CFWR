/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        return null;

    int @MinLen(1) [] y = new int[x];
    @IntRange(from = 1) int z = x;
    @Positive int q = x;
  }

  @SuppressWarnings("index")
  void foo(int x) {
    test(x);
  }

  void foo2(int x) {
    // :: error: (argument)
    test(x);
  }

  void test_lub1(boolean flag, @Positive int x, @IntRange(from = 6, to = 25) int y) {
    int z;
    if (flag) {
      z = x;
    } else {
      z = y;
    }
    @Positive int q = z;
    @IntRange(from = 1) int w = z;
  }

  void test_lub2(boolean flag, @Positive int x, @IntRange(from = -1, to = 11) int y) {
    int z;
    if (flag) {
      z = x;
    } else {
      z = y;
    }
    // :: error: (assignment)
    @Positive int q = z;
    @IntRange(from = -1) int w = z;
  }

  @Positive int id(@Positive int x) {
    return x;
  }

  void test_id(int param) {
    @Positive int x = id(5);
    @IntRange(from = 1) int y = id(5);

    int @MinLen(1) [] a = new int[id(100)];
    // :: error: (assignment)
    int @MinLen(10) [] c = new int[id(100)];

    int q = id(10);

    if (param == q) {
      int @MinLen(1) [] d = new int[param];
    }
      protected int __cfwr_helper107(float __cfwr_p0, boolean __cfwr_p1) {
        while (false) {
            return ((332L & -716) % -449);
            break; // Prevent infinite loops
        }
        while ((null & null)) {
            try {
            while (false) {
            while ((33.87f + '8')) {
            try {
            long __cfwr_val92 = -145L;
        } catch (Exception __cfwr_e32) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e35) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return -371;
    }
    private static String __cfwr_temp47(char __cfwr_p0) {
        Integer __cfwr_entry62 = null;
        while ((91.51 | null)) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 3; __cfwr_i45++) {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 7; __cfwr_i66++) {
            try {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 9; __cfwr_i50++) {
            for (int __cfwr_i4 = 0; __cfwr_i4 < 7; __cfwr_i4++) {
            while ((null << 94.64)) {
            while (((false & null) + ('O' ^ 1.74))) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 1; __cfwr_i59++) {
            for (int __cfwr_i16 = 0; __cfwr_i16 < 5; __cfwr_i16++) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 6; __cfwr_i5++) {
            return null;
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        return "result78";
    }
    protected static double __cfwr_process107(float __cfwr_p0, byte __cfwr_p1, Boolean __cfwr_p2) {
        return null;
        for (int __cfwr_i21 = 0; __cfwr_i21 < 3; __cfwr_i21++) {
            try {
            try {
            byte __cfwr_obj83 = (null - (605L + null));
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        }
        String __cfwr_node80 = "temp78";
        return (false - 99.76f);
    }
}
