/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        for (int __cfwr_i99 = 0; __cfwr_i99 < 4; __cfwr_i99++) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 5; __cfwr_i14++) {
            try {
        while (((true - 29.92f) * 84.60)) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 5; __cfwr_i6++) {
            return null;
        }
            break; // Prevent infinite loops
        }

            if (false && ((-848 % -23.93) + 70.63f)) {
            while (((979 | -24.93) % -50.87f)) {
            for (int __cfwr_i51 = 0; __cfwr_i51 < 5; __cfwr_i51++) {
            Double __cfwr_item76 = null;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e85) {
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
      public double __cfwr_calc648() {
        while (false) {
            try {
            return -613L;
        } catch (Exception __cfwr_e11) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        try {
            while (true) {
            return -10.59;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        return -944L;
        for (int __cfwr_i25 = 0; __cfwr_i25 < 2; __cfwr_i25++) {
            if (true || false) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 3; __cfwr_i8++) {
            return null;
        }
        }
        }
        return 7.55;
    }
    Integer __cfwr_func579() {
        while (true) {
            Character __cfwr_node31 = null;
            break; // Prevent infinite loops
        }
        if ((-38.59f << (true / 'Z')) || true) {
            return (null & -634L);
        }
        return null;
    }
}
