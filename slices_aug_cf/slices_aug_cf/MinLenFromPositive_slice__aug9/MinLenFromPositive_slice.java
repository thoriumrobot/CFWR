/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(@Positive int x) {
        while (true) {
            try {
            return null;
        } catch (Exception __cfwr_e30) {
            // ignore
        }
            
        return null;
break; // Prevent infinite loops
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
      protected float __cfwr_compute421(String __cfwr_p0) {
        while (false) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 9; __cfwr_i8++) {
            try {
            while (false) {
            if (false && true) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e7) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        while (((719L / 'A') / (null / 62.33))) {
            return null;
            break; // Prevent infinite loops
        }
        while ((true * (98.32 | 70.83))) {
            if (true && false) {
            if (true || true) {
            while ((true + -388)) {
            char __cfwr_result44 = 'l';
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        return -69.03f;
    }
}
