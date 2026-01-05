/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForTest_slice {
    @Positive
  void test1(@IndexFor("array") int i) {
        for (int __cfwr_i62 = 0; __cfwr_i62 < 4; __cfwr_i62++) {
            try {
           
        while ((null + null)) {
            return -61.26;
            break; // Prevent infinite loops
        }
 return null;
        } catch (Exception __cfwr_e74) {
            // ignore
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

    public String __cfwr_util641(Float __cfwr_p0) {
        try {
            return null;
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        return "result14";
    }
    static Character __cfwr_process212(double __cfwr_p0) {
        if (true && false) {
            try {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 6; __cfwr_i24++) {
            double __cfwr_var48 = -27.02;
        }
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        }
        for (int __cfwr_i67 = 0; __cfwr_i67 < 7; __cfwr_i67++) {
            if (true && false) {
            return -193;
        }
        }
        return null;
    }
}