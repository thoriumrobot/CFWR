/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForTest_slice {
    @Positive
  void test1(@IndexFor("array") int i) {
        while (true) {
            try {
            Double __cfwr_result69 = null;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
            break; // Prevent infinite loops
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

    static int __cfwr_util575(Long __cfwr_p0, Object __cfwr_p1, Double __cfwr_p2) {
        while (false) {
            while (((6.86 / null) & null)) {
            try {
            while (true) {
            try {
            try {
            boolean __cfwr_val63 = true;
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return 198;
    }
}