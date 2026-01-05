/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForTest_slice {
    @Positive
  void test1(@IndexFor("array") int i) {
        Character __cfwr_entry29 = null;

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

    public static boolean __cfwr_proc645(String __cfwr_p0, int __cfwr_p1) {
        if (false && true) {
            double __cfwr_item8 = 51.46;
        }
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return "result11";
        return null;
        return false;
    }
}