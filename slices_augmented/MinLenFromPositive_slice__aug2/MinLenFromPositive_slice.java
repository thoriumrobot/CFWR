/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class MinLenFromPositive_slice {
    @Positive
  void test(@Positive int x) {
        while (true) {
            Boolean __cfwr_entry73 = null;
            break; // Prevent infinite loops
        }

    @Positive
    int @MinLen(1) [] y = new int[x];
    @Positive
    
        byte __cfwr_node34 = (-677L & 8.64f);
@IntRange(from = 1) int z = x;
    @Positive
    @Positive int q = x;
    @Positive
  }

    @Positive
  void foo(int x) {
    @Positive
    test(x);
    @Positive
  }

    @Positive
  void foo2(int x) {
    // :: error: (argument)
    @Positive
    test(x);
    @Positive
  }

    @Positive
  void test_lub1(boolean flag, @Positive int x, @IntRange(from = 6, to = 25) int y) {
    @Positive
    int z;
    @Positive
    if (flag) {
    @Positive
      z = x;
    @Positive
    } else {
    @Positive
      z = y;
    @Positive
    }
    @Positive
    @Positive int q = z;
    @Positive
    @IntRange(from = 1) int w = z;
    @Positive
  }

    @Positive
  void test_lub2(boolean flag, @Positive int x, @IntRange(from = -1, to = 11) int y) {
    @Positive
    int z;
    @Positive
    if (flag) {
    @Positive
      z = x;
    @Positive
    } else {
    @Positive
      z = y;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @Positive int q = z;
    @Positive
    @IntRange(from = -1) int w = z;
    @Positive
  }

    static char __cfwr_calc3(String __cfwr_p0, Boolean __cfwr_p1, Object __cfwr_p2) {
        return null;
        for (int __cfwr_i28 = 0; __cfwr_i28 < 1; __cfwr_i28++) {
            try {
            return -883L;
        } catch (Exception __cfwr_e7) {
            // ignore
        }
        }
        try {
            while (true) {
            double __cfwr_item9 = 35.04;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        while ((-16.16f >> 70)) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 4; __cfwr_i34++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        return 'Q';
    }
}