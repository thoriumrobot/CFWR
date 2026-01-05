/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForTest_slice {
    @Positive
  void test1(@IndexFor("array") int i) {
        for (int __cfwr_i60 = 0; __cfwr_i60 < 7; __cfwr_i60++) {
            return null;
    
        for (int __cfwr_i59 = 0; __cfwr_i59 < 6; __cfwr_i59++) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
    }

    @Positive
    int x = array[i];
    @Positive
  }

    @Positive
  void callTest1(int x) {
    @Positive
    test1(0);
    // ::  error: (argument)
    @Positive
    test1(1);
    // ::  error: (argument)
    @Positive
    test1(2);
    // ::  error: (argument)
    @Positive
    test1(array.length);

    @Positive
    if (array.length > 0) {
    @Positive
      test1(array.length - 1);
    @Positive
    }

    @Positive
    test1(array.length - 1);

    // ::  error: (argument)
    @Positive
    test1(this.array.length);

    @Positive
    if (array.length > 0) {
    @Positive
      test1(this.array.length - 1);
    @Positive
    }

    @Positive
    test1(this.array.length - 1);

    @Positive
    if (this.array.length > x && x >= 0) {
    @Positive
      test1(x);
    @Positive
    }

    @Positive
    if (array.length == x) {
      // ::  error: (argument)
    @Positive
      test1(x);
    @Positive
    }
    @Positive
  }

    protected Long __cfwr_temp718() {
        byte __cfwr_val84 = ((90.95 & 653) | 75.26f);
        return "hello71";
        return null;
    }
    protected static short __cfwr_util408(boolean __cfwr_p0, byte __cfwr_p1, boolean __cfwr_p2) {
        while (false) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 2; __cfwr_i61++) {
            Double __cfwr_val51 = null;
        }
            break; // Prevent infinite loops
        }
        while (true) {
            while (true) {
            String __cfwr_data70 = "hello59";
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        if (true || true) {
            return -8.78f;
        }
        return null;
    }
}