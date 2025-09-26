/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        for (int __cfwr_i54 = 0; __cfwr_i54 < 7; __cfwr_i54++) {
            while (true) {
            if (true || false) {
            try {
            for (int __cfwr_i22 = 0; __cfwr_i22 < 8; __cfwr_i22++) {
            while (false) {
            for (int __cfwr_i68 = 0; __cfwr_i68 < 9; __cfwr_i68++) {
            char __cfwr_result27 = 'O';
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        }
            break; // Prevent infinite loops
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
      private Character __cfwr_process436() {
        if (((873 << 1.19) - 91.00) || false) {
            return null;
        }
        return null;
    }
    short __cfwr_aux170(boolean __cfwr_p0) {
        return null;
        return null;
    }
}
