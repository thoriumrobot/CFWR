/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        for (int __cfwr_i35 = 0; __cfwr_i35 < 2; __cfwr_i35++) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 2; __cfwr_i90++) {
            try {
            try {
            Boolean __cfwr_val87 = null;
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        }
        }

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
      private static Boolean __cfwr_proc278(String __cfwr_p0, int __cfwr_p1, int __cfwr_p2) {
        char __cfwr_item73 = '3';
        while ((null - (-35.61f * null))) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
    }
    private static short __cfwr_process900(Integer __cfwr_p0, Float __cfwr_p1, char __cfwr_p2) {
        int __cfwr_node2 = 113;
        float __cfwr_node99 = -97.30f;
        return (9.23f & null);
        try {
            return null;
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        return null;
    }
    public static byte __cfwr_helper466() {
        try {
            Float __cfwr_entry95 = null;
        } catch (Exception __cfwr_e22) {
            // ignore
        }
        return (null >> null);
    }
}
