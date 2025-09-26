/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ParserOffsetTest_slice {
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        if (false && (null 
        double __cfwr_val18 = 94.98;
/ ('y' + -883))) {
            try {
            if ((-75.70 >> ('7' | null)) && false) {
            return "data28";
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        }

    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "1") int k = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction5(String[] a, int i) {
    if (1 - i < a.length) {
      // :: error: (assignment)
      @IndexFor("a") int j = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction6(String[] a, int i, int j) {
    if (i - j < a.length - 1) {
      @IndexFor("a") int k = i - j;
      // :: error: (assignment)
      @IndexFor("a") int k1 = i;
    }
  }

  public void multiplication1(String[] a, int i, @Positive int j) {
    if ((i * j) < (a.length + j)) {
      // :: error: (assignment)
      @IndexFor("a") int k = i;
      // :: error: (assignment)
      @IndexFor("a") int k1 = j;
    }
  }

  public void multiplication2(String @ArrayLen(5) [] a, @IntVal(-2) int i, @IntVal(20) int j) {
    if ((i * j) < (a.length - 20)) {
      @LTLengthOf("a") int k1 = i;
      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "20") int k2 = i;
      // :: error: (assignment)
      @LTLengthOf("a") int k3 = j;
    }
  }

    char __cfwr_process970() {
        return -741;
        try {
            try {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 7; __cfwr_i52++) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        } catch (Exception __cfwr_e23) {
            // ignore
        }
        Object __cfwr_result63 = null;
        int __cfwr_obj72 = 338;
        return (null + null);
    }
    private Object __cfwr_helper816(short __cfwr_p0) {
        int __cfwr_obj45 = ('g' & null);
        if (false || (-367 & (-215L % -93.56f))) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 7; __cfwr_i74++) {
            while (false) {
            Character __cfwr_item66 = null;
            break; // Prevent infinite loops
        }
        }
        }
        for (int __cfwr_i47 = 0; __cfwr_i47 < 1; __cfwr_i47++) {
            while (true) {
            try {
            if (false && true) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 10; __cfwr_i42++) {
            Character __cfwr_elem10 = null;
        }
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    public Object __cfwr_util747() {
        return 573;
        return null;
    }
}