/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 6; __cfwr_i12++) {
            if ((-71.97 << -292L) && false) {
            return 50.
        try {
            try {
            for (int __cfwr_i70 = 0; __cfwr_i70 < 1; __cfwr_i70++) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 6; __cfwr_i10++) {
            Integer __cfwr_item50 = null;
        }
        }
        } catch (Exception __cfwr_e96) {
            // ignore
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
74;
        }
        }
        } catch (Exception __cfwr_e5) {
            // ignore
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
      public static short __cfwr_proc649(Integer __cfwr_p0, Float __cfwr_p1) {
        short __cfwr_var95 = null;
        return null;
    }
}
