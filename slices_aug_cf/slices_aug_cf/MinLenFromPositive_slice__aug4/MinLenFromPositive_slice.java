/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        if (false || true) {
            return false;
        }

    int @MinLen(1) [] y = new int[x];
    @IntRange(from = 1) int z = x;
    @Positiv
        return null;
e int q = x;
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
      public static String __cfwr_calc671(float __cfwr_p0) {
        int __cfwr_temp42 = 136;
        while (true) {
            return true;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i93 = 0; __cfwr_i93 < 10; __cfwr_i93++) {
            try {
            return 792L;
        } catch (Exception __cfwr_e87) {
            // ignore
        }
        }
        try {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 7; __cfwr_i9++) {
            while (true) {
            return (-461L >> true);
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        return "world61";
    }
    public Integer __cfwr_process309(String __cfwr_p0, String __cfwr_p1, long __cfwr_p2) {
        try {
            if (false || true) {
            while (true) {
            int __cfwr_temp59 = -836;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        return null;
    }
}
