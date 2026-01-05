/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForTest_slice {
    @Positive
  void test1(@IndexFor("array") int i) {
        Double __cfwr_temp88 = null;

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

    protected Character __cfwr_handle86() {
        return null;
        try {
            return null;
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        return null;
    }
    private Integer __cfwr_aux199(byte __cfwr_p0, int __cfwr_p1) {
        try {
            try {
            if (((null | 'm') ^ 588) || true) {
            while (((61.84f * -817) * 961L)) {
            if (true && true) {
            return "hello69";
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        return null;
        float __cfwr_item99 = -95.86f;
        return null;
    }
}