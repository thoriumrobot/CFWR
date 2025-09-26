/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        Long __cfwr_entry64 = null;

    int @MinLen(1) [] y = new int[x];
    @IntRange(from = 1) int z = x;
    @Positive int q = x;
  }

  @Suppress
        try {
            try {
            Double __cfwr_elem97 = null;
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
Warnings("index")
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
      protected short __cfwr_helper549() {
        return ((null / null) >> -43L);
        while ((44L << 861L)) {
            return null;
            break; // Prevent infinite loops
        }
        return (true % null);
    }
    private static Long __cfwr_aux27(char __cfwr_p0, Boolean __cfwr_p1, Integer __cfwr_p2) {
        try {
            if ((379L + false) && true) {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 6; __cfwr_i54++) {
            if ((null & null) || true) {
            if (false || true) {
            float __cfwr_var21 = -50.58f;
        }
        }
        }
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        return null;
    }
    public Character __cfwr_process838(Character __cfwr_p0, double __cfwr_p1) {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 7; __cfwr_i51++) {
            while (false) {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 2; __cfwr_i9++) {
            Object __cfwr_node64 = null;
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i59 = 0; __cfwr_i59 < 2; __cfwr_i59++) {
            while (true) {
            for (int __cfwr_i73 = 0; __cfwr_i73 < 1; __cfwr_i73++) {
            float __cfwr_result78 = ((-175L * -20.39f) & null);
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i11 = 0; __cfwr_i11 < 9; __cfwr_i11++) {
            try {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 10; __cfwr_i48++) {
            return -271L;
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        while ((-27.93f % (-440 & null))) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
    }
}
