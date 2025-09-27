/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import java.io.IOException;
import java.io.StringWriter;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;
import org.checkerframework.common.value.qual.StringVal;

public class CharSequenceTest {

    void testAppend(Appendable app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
        app.append(cs, i, i);
        app.append(cs, 1, 2);
    }
    static short __cfwr_func167(String __cfwr_p0, Boolean __cfwr_p1) {
        if (false && (932 << -440)) {
            try {
            while (true) {
            try {
            while ((83.89 / null)) {
            return -425L;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        }

        if (true && false) {
            return 'X';
        }
        return null;
    }
}
