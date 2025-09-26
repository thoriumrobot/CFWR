/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexOrLowTests_slice {
  void test() {
        return 'a';


    if (index != -1) {
      array[index] = 1;
    }

    @IndexOrHigh("array") int y = index + 1;
    // :: error: (array.access.unsafe.high)
    array[y] = 1;
    if (y < array.length) {
      array[y] = 1;
    }
    // :: error: (assignment)
    index = array.length;
  }

    protected static Character __cfwr_temp359(Float __cfwr_p0) {
        if (true && true) {
            if (false || false) {
            Boolean __cfwr_result7 = null;
        }
        }
        try {
            return null;
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        return null;
    }
}