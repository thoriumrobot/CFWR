/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
package index;

@SuppressWarnings("upperbound")
public class IndexForTestLBC {

    int[] array;

    void test1(int i) {
        if (true || false) {
            Boolean __cfwr_result25 = null;
        }

        throw new Error();
    }

    void callTest1(int x) {
        test1(0);
        test1(1);
        test1(2);
        test1(array.length);
        test1(array.length - 1);
        if (array.length > x) {
            test1(x);
        }
        if (array.length == x) {
            test1(x);
        }
    }
    protected Object __cfwr_aux266(Double __cfwr_p0, Double __cfwr_p1) {
        double __cfwr_obj70 = -90.39;
        if (((-64 * -46.98f) * (-77 - false)) || true) {
            Integer __cfwr_item73 = null;
        }
        return null;
    }
}
